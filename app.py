import os
import io
import json
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, jsonify, flash
)
from werkzeug.utils import secure_filename

# Optional Pillow fallback for reading problematic JPEGs (e.g., CMYK)
try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None  # Pillow not installed; fallback will be skipped

from utils.filtering import median_filter_enhancement
from utils.adaptive_ahe import (
    standard_clahe,
    clahe_bilateral,
    clahe_unsharp_masking,
    hybrid_adaptive_hist_eq
)
from utils.metrics import compute_all_metrics
from utils.report_generator import generate_pdf_report

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
REPORTS_FOLDER = os.path.join(BASE_DIR, "static", "reports")

# Use-case presets for quick parameter tuning
PRESETS: Dict[str, Dict[str, Any]] = {
    "general": {"clip_limit": 2.0, "block_size": 8, "kernel_size": 5},
    "forensic": {
        "clip_limit": 3.0, "block_size": 4,
        "bilateral_d": 9, "sigma_color": 100, "sigma_space": 100
    },
    "currency": {"clip_limit": 2.5, "block_size": 4, "unsharp_amount": 2.0},
    "medical": {
        "clip_limit": 1.5, "block_size": 8,
        "bilateral_d": 7, "sigma_color": 50, "sigma_space": 50
    }
}

# -----------------------------------------------------------------------------
# App Factory
# -----------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["RESULTS_FOLDER"] = RESULTS_FOLDER
    app.config["REPORTS_FOLDER"] = REPORTS_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

    # Ensure directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _decode_from_bytes(data: bytes) -> Optional[np.ndarray]:
        """
        Try decoding image bytes to RGB using OpenCV, fallback to Pillow if available.
        Returns None if all attempts fail.
        """
        if not data:
            return None
        # First attempt: OpenCV imdecode
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        # Fallback: Pillow (handles CMYK JPEGs and some corner cases)
        if PILImage is not None:
            try:
                with PILImage.open(io.BytesIO(data)) as im:
                    im = im.convert("RGB")
                    return np.array(im)
            except Exception:
                pass
        return None

    def cv2_imread_rgb(path: str, retries: int = 6, delay: float = 0.25) -> np.ndarray:
        """
        Robust image reader (Windows/OneDrive safe):
        - Try np.fromfile + cv2.imdecode
        - Retry a few times (handles transient OneDrive/AV locks)
        - Fallback to Pillow if available
        """
        last_err = None
        for i in range(max(1, retries)):
            try:
                data = np.fromfile(path, dtype=np.uint8)
                if data.size > 0:
                    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                last_err = e
            # Pillow fallback
            if PILImage is not None:
                try:
                    with PILImage.open(path) as im:
                        im = im.convert("RGB")
                        return np.array(im)
                except Exception as e:
                    last_err = e
            # Wait and retry in case of sync/lock
            time.sleep(delay)
        raise ValueError(
            f"Failed to read image at path: {path}. "
            "Possible causes: zero-byte upload, locked by OneDrive/AV, or unsupported/corrupted file."
            + (f" Last error: {last_err}" if last_err else "")
        )

    def cv2_imsave_rgb(path: str, rgb_image: np.ndarray) -> None:
        """
        Robust image writer using imencode + tofile for Windows path safety.
        Falls back to cv2.imwrite if needed.
        """
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        ext = os.path.splitext(path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            ext = ".jpg"
            path = path + ext
        try:
            ok, buf = cv2.imencode(ext, bgr)
            if not ok:
                raise ValueError("imencode failed")
            buf.tofile(path)
        except Exception:
            cv2.imwrite(path, bgr)

    def ensure_odd(n: int) -> int:
        """Ensure kernel size is odd and >= 1."""
        n = max(1, int(n))
        return n if n % 2 == 1 else n + 1

    def apply_presets(base_params: Dict[str, Any], use_case: str) -> Dict[str, Any]:
        """Overlay preset parameters for a given use_case onto base_params."""
        params = dict(base_params)
        preset = PRESETS.get(use_case)
        if preset:
            params.update({k: v for k, v in preset.items() if v is not None})
        return params

    def parse_params(form: Dict[str, str]) -> Dict[str, Any]:
        """Parse parameters from form with sensible defaults and type casting."""
        def get_float(name: str, default: float) -> float:
            try:
                return float(form.get(name, default))
            except Exception:
                return default

        def get_int(name: str, default: int) -> int:
            try:
                return int(float(form.get(name, default)))
            except Exception:
                return default

        params = {
            "clip_limit": get_float("clip_limit", 2.0),
            "block_size": get_int("block_size", 8),
            "kernel_size": ensure_odd(get_int("kernel_size", 5)),
            "bilateral_d": get_int("bilateral_d", 9),
            "sigma_color": get_float("sigma_color", 75.0),
            "sigma_space": get_float("sigma_space", 75.0),
            "unsharp_amount": get_float("unsharp_amount", 1.5),
            "unsharp_radius": get_float("unsharp_radius", 1.0),
            "unsharp_threshold": get_int("unsharp_threshold", 0),
        }
        return params

    def process_image(mode: str, rgb_image: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """Process image according to selected mode and parameters."""
        block_size = max(2, int(params.get("block_size", 8)))
        grid = (block_size, block_size)
        clip_limit = float(params.get("clip_limit", 2.0))

        if mode == "median":
            k = ensure_odd(int(params.get("kernel_size", 5)))
            out = median_filter_enhancement(rgb_image, k)
            method_used = f"Median (k={k})"

        elif mode == "clahe":
            out = standard_clahe(rgb_image, clip_limit=clip_limit, tile_grid_size=grid)
            method_used = f"CLAHE (clip={clip_limit}, grid={block_size}x{block_size})"

        elif mode == "bilateral":
            out = clahe_bilateral(
                rgb_image,
                clip_limit=clip_limit,
                tile_grid_size=grid,
                bilateral_d=int(params.get("bilateral_d", 9)),
                sigma_color=float(params.get("sigma_color", 75.0)),
                sigma_space=float(params.get("sigma_space", 75.0))
            )
            method_used = (
                f"CLAHE+Bilateral (clip={clip_limit}, grid={block_size}x{block_size}, "
                f"d={params.get('bilateral_d', 9)}, sc={params.get('sigma_color', 75)}, ss={params.get('sigma_space', 75)})"
            )

        elif mode == "unsharp":
            out = clahe_unsharp_masking(
                rgb_image,
                clip_limit=clip_limit,
                tile_grid_size=grid,
                amount=float(params.get("unsharp_amount", 1.5)),
                radius=float(params.get("unsharp_radius", 1.0)),
                threshold=int(params.get("unsharp_threshold", 0))
            )
            method_used = (
                f"CLAHE+Unsharp (clip={clip_limit}, grid={block_size}x{block_size}, "
                f"amount={params.get('unsharp_amount', 1.5)}, radius={params.get('unsharp_radius', 1.0)}, "
                f"th={params.get('unsharp_threshold', 0)})"
            )
        else:
            out = hybrid_adaptive_hist_eq(rgb_image, block_size=block_size, limit=clip_limit, mode="standard")
            method_used = f"Hybrid CLAHE (auto, grid={block_size}x{block_size})"

        return out, method_used

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", presets=list(PRESETS.keys()))

    @app.route("/upload", methods=["POST"])
    def upload():
        # Validate file
        if "image" not in request.files:
            flash("No file part in the request.")
            return redirect(url_for("index"))
        file = request.files["image"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Unsupported file type. Allowed: .jpg, .jpeg, .png")
            return redirect(url_for("index"))

        # Parse form fields
        mode = request.form.get("mode", "clahe")  # median | clahe | bilateral | unsharp
        use_case = request.form.get("use_case", "general")
        base_params = parse_params(request.form)
        params = apply_presets(base_params, use_case)

        # Prepare filename + path
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        ts = timestamp()
        server_filename = f"{name}_{ts}{ext.lower()}"
        upload_path = os.path.join(UPLOAD_FOLDER, server_filename)

        # Read file bytes first (avoid immediate OneDrive readback)
        file_bytes = file.read()
        if not file_bytes:
            flash("Uploaded file is empty. Please try again.")
            return redirect(url_for("index"))

        # Decode from bytes directly
        rgb = _decode_from_bytes(file_bytes)
        if rgb is None:
            flash("Failed to decode image. Ensure it's a valid JPG/PNG (HEIC not supported).")
            return redirect(url_for("index"))

        # Save original bytes to disk using binary write (avoid file.save + potential locks)
        try:
            with open(upload_path, "wb") as f:
                f.write(file_bytes)
        except Exception as e:
            flash(f"Could not save uploaded file: {e}")
            return redirect(url_for("index"))

        # Process
        try:
            enhanced_rgb, method_used = process_image(mode, rgb, params)
        except Exception as e:
            flash(f"Processing error: {e}")
            return redirect(url_for("index"))

        # Save enhanced
        enhanced_filename = f"{name}_{mode}_{ts}.jpg"
        enhanced_path = os.path.join(RESULTS_FOLDER, enhanced_filename)
        cv2_imsave_rgb(enhanced_path, enhanced_rgb)

        # Compute metrics
        metrics = compute_all_metrics(rgb, enhanced_rgb)

        # Session history
        result_record = {
            "original": os.path.relpath(upload_path, BASE_DIR).replace("\\", "/"),
            "enhanced": os.path.relpath(enhanced_path, BASE_DIR).replace("\\", "/"),
            "metrics": metrics,
            "mode": mode,
            "method_used": method_used,
            "params": params,
            "use_case": use_case,
            "timestamp": ts,
            "filename": server_filename,
            "enhanced_filename": enhanced_filename
        }
        session["last_result"] = result_record
        history = session.get("history", [])
        history.insert(0, result_record)
        session["history"] = history

        return redirect(url_for("result"))

    @app.route("/result", methods=["GET"])
    def result():
        data = session.get("last_result")
        if not data:
            flash("No result found. Please upload an image.")
            return redirect(url_for("index"))

        metrics = data.get("metrics", {})
        metrics_labels = ["PSNR", "MSE", "SSIM"]
        metrics_values = [metrics.get("psnr", 0), metrics.get("mse", 0), metrics.get("ssim", 0)]

        return render_template(
            "result.html",
            data=data,
            metrics_labels=json.dumps(metrics_labels),
            metrics_values=json.dumps(metrics_values)
        )

    @app.route("/gallery", methods=["GET"])
    def gallery():
        history = session.get("history", [])
        return render_template("gallery.html", history=history)

    @app.route("/dashboard", methods=["GET"])
    def dashboard():
        history = session.get("history", [])

        methods = ["median", "clahe", "bilateral", "unsharp"]
        agg = {m: {"psnr": [], "mse": [], "ssim": []} for m in methods}
        for rec in history:
            m = rec.get("mode")
            met = rec.get("metrics", {})
            if m in agg:
                if "psnr" in met:
                    agg[m]["psnr"].append(met["psnr"])
                if "mse" in met:
                    agg[m]["mse"].append(met["mse"])
                if "ssim" in met:
                    agg[m]["ssim"].append(met["ssim"])

        avg = {}
        for m in methods:
            if agg[m]["psnr"]:
                avg[m] = {
                    "psnr": float(np.mean(agg[m]["psnr"])),
                    "mse": float(np.mean(agg[m]["mse"])) if agg[m]["mse"] else 0.0,
                    "ssim": float(np.mean(agg[m]["ssim"])) if agg[m]["ssim"] else 0.0
                }
            else:
                avg[m] = {"psnr": 0.0, "mse": 0.0, "ssim": 0.0}

        best = {
            "psnr": max(avg.items(), key=lambda kv: kv[1]["psnr"])[0] if avg else "-",
            "mse": min(avg.items(), key=lambda kv: kv[1]["mse"])[0] if avg else "-",
            "ssim": max(avg.items(), key=lambda kv: kv[1]["ssim"])[0] if avg else "-",
        }

        plotly_data = {
            "methods": methods,
            "psnr": [avg[m]["psnr"] for m in methods],
            "mse": [avg[m]["mse"] for m in methods],
            "ssim": [avg[m]["ssim"] for m in methods]
        }

        return render_template(
            "dashboard.html",
            avg=avg, best=best, history=history, plotly_data=json.dumps(plotly_data)
        )

    @app.route("/generate_report/<enhanced_filename>", methods=["GET"])
    def generate_report(enhanced_filename: str):
        history = session.get("history", [])
        record = next((r for r in history if r.get("enhanced_filename") == enhanced_filename), None)
        if not record:
            flash("Record not found for report generation.")
            return redirect(url_for("gallery"))

        original_path = os.path.join(BASE_DIR, record["original"])
        enhanced_path = os.path.join(BASE_DIR, record["enhanced"])
        metrics = record["metrics"]
        params = {
            "mode": record["mode"],
            "method_used": record["method_used"],
            "use_case": record["use_case"],
            **record.get("params", {})
        }
        out_name = f"report_{record['filename'].rsplit('.', 1)[0]}_{record['mode']}.pdf"
        out_path = os.path.join(REPORTS_FOLDER, out_name)

        try:
            pdf_path = generate_pdf_report(
                original_path=original_path,
                enhanced_path=enhanced_path,
                metrics=metrics,
                parameters=params,
                output_path=out_path
            )
        except Exception as e:
            flash(f"Report generation failed: {e}")
            return redirect(url_for("result"))

        return send_file(pdf_path, as_attachment=True)

    @app.route("/adjust_parameters", methods=["POST"])
    def adjust_parameters():
        """AJAX endpoint: re-process with new parameters and return metrics + image."""
        data = request.get_json(force=True, silent=True) or {}
        record = session.get("last_result")
        if not record:
            return jsonify({"error": "No active image to adjust."}), 400

        mode = data.get("mode", record.get("mode", "clahe"))
        params = data.get("params", {})
        prev_params = record.get("params", {})
        merged_params = {**prev_params, **params}

        # Read original with robust reader + retries
        try:
            rgb = cv2_imread_rgb(os.path.join(BASE_DIR, record["original"]))
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        try:
            enhanced_rgb, method_used = process_image(mode, rgb, merged_params)
        except Exception as e:
            return jsonify({"error": f"Processing error: {e}"}), 400

        base_name = record["filename"].rsplit(".", 1)[0]
        ts = timestamp()
        new_filename = f"{base_name}_{mode}_{ts}.jpg"
        new_path = os.path.join(RESULTS_FOLDER, new_filename)
        cv2_imsave_rgb(new_path, enhanced_rgb)

        metrics = compute_all_metrics(rgb, enhanced_rgb)

        new_record = dict(record)
        new_record.update({
            "enhanced": os.path.relpath(new_path, BASE_DIR).replace("\\", "/"),
            "enhanced_filename": new_filename,
            "metrics": metrics,
            "params": merged_params,
            "mode": mode,
            "method_used": method_used,
            "timestamp": ts
        })
        session["last_result"] = new_record

        return jsonify({
            "enhanced": url_for("static", filename=f"results/{new_filename}"),
            "metrics": metrics,
            "mode": mode,
            "method_used": method_used
        })

    return app


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)