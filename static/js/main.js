(function () {
  // Persisted theme
  const html = document.documentElement;
  const stored = localStorage.getItem('theme');
  if (stored) html.setAttribute('data-theme', stored);

  document.getElementById('themeToggle')?.addEventListener('click', () => {
    const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  });

  // Upload UI helpers
  const imageInput = document.getElementById('image');
  const previewImg = document.getElementById('previewImage');
  const dropZone = document.getElementById('dropZone');
  const form = document.getElementById('uploadForm');

  if (imageInput && previewImg) {
    imageInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        previewImg.src = ev.target.result;
        previewImg.classList.remove('d-none');
      };
      reader.readAsDataURL(file);
    });
  }

  if (form) {
    const submitSpinner = document.getElementById('submitSpinner');
    form.addEventListener('submit', () => {
      submitSpinner?.classList.remove('d-none');
    });
  }

  // Live value labels on range inputs (index page)
  const clipLimit = document.getElementById('clip_limit');
  const clipLimitVal = document.getElementById('clipLimitVal');
  clipLimit?.addEventListener('input', () => clipLimitVal.textContent = clipLimit.value);

  const blockSize = document.getElementById('block_size');
  const blockSizeVal = document.getElementById('blockSizeVal');
  blockSize?.addEventListener('input', () => blockSizeVal.textContent = blockSize.value);

  const kernelSize = document.getElementById('kernel_size');
  const kernelSizeVal = document.getElementById('kernelSizeVal');
  kernelSize?.addEventListener('input', () => kernelSizeVal.textContent = kernelSize.value);

  // Use-case preset auto-fill on index page
  const useCase = document.getElementById('use_case');
  useCase?.addEventListener('change', () => {
    const preset = useCase.value;
    const presets = {
      general: {clip_limit: 2.0, block_size: 8, kernel_size: 5},
      forensic: {clip_limit: 3.0, block_size: 4},
      currency: {clip_limit: 2.5, block_size: 4},
      medical: {clip_limit: 1.5, block_size: 8}
    };
    const p = presets[preset];
    if (!p) return;
    if (clipLimit && p.clip_limit) { clipLimit.value = p.clip_limit; clipLimitVal.textContent = p.clip_limit; }
    if (blockSize && p.block_size) { blockSize.value = p.block_size; blockSizeVal.textContent = p.block_size; }
    if (kernelSize && p.kernel_size) { kernelSize.value = p.kernel_size; kernelSizeVal.textContent = p.kernel_size; }
  });

  // Gallery filter & sort
  const filterMode = document.getElementById('filterMode');
  const sortBy = document.getElementById('sortBy');
  const grid = document.getElementById('galleryGrid');

  function refreshGallery() {
    if (!grid) return;
    const items = Array.from(grid.querySelectorAll('.gallery-item'));
    const mode = filterMode?.value || 'all';
    items.forEach(item => {
      const m = item.getAttribute('data-mode');
      item.style.display = (mode === 'all' || mode === m) ? '' : 'none';
    });

    const visible = items.filter(i => i.style.display !== 'none');
    const key = sortBy?.value || 'date';
    visible.sort((a, b) => {
      if (key === 'name') {
        return (a.dataset.name || '').localeCompare(b.dataset.name || '');
      } else if (key === 'psnr') {
        return (parseFloat(b.dataset.psnr || '0') - parseFloat(a.dataset.psnr || '0'));
      } else { // date
        return (b.dataset.date || '').localeCompare(a.dataset.date || '');
      }
    });
    visible.forEach(i => grid.appendChild(i));
  }

  filterMode?.addEventListener('change', refreshGallery);
  sortBy?.addEventListener('change', refreshGallery);
  refreshGallery();

  // Result page: Adjust parameters AJAX
  const applyAdjust = document.getElementById('applyAdjust');
  const enhancedImage = document.getElementById('enhancedImage');
  const adjustSpinner = document.getElementById('adjustSpinner');

  function getVal(id) {
    const el = document.getElementById(id);
    return el ? el.value : null;
  }

  applyAdjust?.addEventListener('click', async () => {
    adjustSpinner?.classList.remove('d-none');
    try {
      const payload = {
        mode: getVal('adj_mode'),
        params: {
          clip_limit: parseFloat(getVal('adj_clip')),
          block_size: parseInt(getVal('adj_block')),
          kernel_size: parseInt(getVal('adj_kernel')),
          bilateral_d: parseInt(getVal('adj_bilat_d')),
          sigma_color: parseFloat(getVal('adj_sigma_color')),
          sigma_space: parseFloat(getVal('adj_sigma_space')),
          unsharp_amount: parseFloat(getVal('adj_amount')),
          unsharp_radius: parseFloat(getVal('adj_radius')),
          unsharp_threshold: parseInt(getVal('adj_threshold')),
        }
      };
      const res = await fetch('/adjust_parameters', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Adjustment failed');

      // Update enhanced image
      if (enhancedImage) {
        // Cache-busting
        enhancedImage.src = data.enhanced + '?t=' + Date.now();
      }

      // Update Plotly chart
      const m = data.metrics || {};
      const vals = [m.psnr || 0, m.mse || 0, m.ssim || 0];
      Plotly.react('metricsChart', [{type: 'bar', x: ['PSNR', 'MSE', 'SSIM'], y: vals, marker: {color: ['#1976D2','#EF5350','#43A047']}}], {title: 'Image Quality Metrics'});

    } catch (err) {
      alert(err.message || 'Error while adjusting parameters.');
    } finally {
      adjustSpinner?.classList.add('d-none');
    }
  });

  // Dashboard export CSV
  const exportBtn = document.getElementById('exportCSV');
  exportBtn?.addEventListener('click', () => {
    const rows = [];
    const table = document.querySelector('table');
    table.querySelectorAll('tr').forEach(tr => {
      const cols = Array.from(tr.querySelectorAll('th,td')).map(td => `"${td.textContent.trim()}"`);
      rows.push(cols.join(','));
    });
    const blob = new Blob([rows.join('\n')], {type: 'text/csv;charset=utf-8;'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'dashboard_metrics.csv';
    a.click();
    URL.revokeObjectURL(a.href);
  });
})();