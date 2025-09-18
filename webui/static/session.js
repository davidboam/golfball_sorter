document.addEventListener('DOMContentLoaded', () => {
  const imgContainer = document.getElementById('images');
  const brandInput = document.getElementById('brand-input');
  const modelInput = document.getElementById('model-input');
  const brandList = document.getElementById('brand-list');
  const modelList = document.getElementById('model-list');
  const toggleAllBtn = document.getElementById('toggle-attn-all');
  const attnHeadSel = document.getElementById('attn-head');
  const suggestBox = document.getElementById('model-suggest');
  let attnAllOn = false;
  const imgEls = [];
  let selectedGrade = '';
  let selectedColor = '';

  function setActive(buttons, value) {
    buttons.forEach(btn => {
      if (btn.dataset.value === value) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  }

  const gradeBtns = Array.from(document.querySelectorAll('.grade-btn'));
  gradeBtns.forEach(btn => btn.addEventListener('click', () => {
    selectedGrade = btn.dataset.value;
    setActive(gradeBtns, selectedGrade);
  }));

  const colorBtns = Array.from(document.querySelectorAll('.color-btn'));
  colorBtns.forEach(btn => btn.addEventListener('click', () => {
    selectedColor = btn.dataset.value;
    setActive(colorBtns, selectedColor);
  }));

  // load last label for sticky defaults
  fetch('/api/last-label').then(r => r.json()).then(data => {
    if (data.brand) brandInput.value = data.brand;
    if (data.model) modelInput.value = data.model;
    if (data.grade) {
      selectedGrade = data.grade; setActive(gradeBtns, selectedGrade);
    }
    if (data.color) {
      selectedColor = data.color; setActive(colorBtns, selectedColor);
    }
  });

  // load session images
  fetch(`/api/session/${SESSION_ID}`).then(r => r.json()).then(data => {
    data.images.forEach(img => {
      const card = document.createElement('div');
      card.className = 'card';
      const im = document.createElement('img');
      im.src = `/captures/${SESSION_ID}/${img}`;
      im.alt = img;
      im.dataset.filename = img;
      im.dataset.overlay = '0';
      card.appendChild(im);
      imgContainer.appendChild(card);
      imgEls.push(im);
    });
  });

  // global toggle attention for all images
  function applyAttentionAll(on) {
    const head = attnHeadSel ? attnHeadSel.value : 'grade';
    imgEls.forEach(im => {
      const img = im.dataset.filename;
      if (on) {
        im.src = `/api/attn/${SESSION_ID}/${img}?head=${encodeURIComponent(head)}`;
        im.dataset.overlay = '1';
      } else {
        im.src = `/captures/${SESSION_ID}/${img}`;
        im.dataset.overlay = '0';
      }
    });
  }
  if (toggleAllBtn) {
    toggleAllBtn.addEventListener('click', () => {
      attnAllOn = !attnAllOn;
      applyAttentionAll(attnAllOn);
    });
  }
  if (attnHeadSel) {
    attnHeadSel.addEventListener('change', () => {
      if (attnAllOn) applyAttentionAll(true);
    });
  }

  // load suggestions
  fetch(`/api/suggest?ball_id=${SESSION_ID}&image_name=ALL`).then(r => {
    if (r.status === 200) {
      r.json().then(d => {
        if (d.brand && !brandInput.value) brandInput.value = d.brand;
        if (d.grade) { selectedGrade = d.grade; setActive(gradeBtns, selectedGrade); }
        // Show model confidence summary text
        const parts = [];
        if (typeof d.brand_conf === 'number' && d.brand) parts.push(`Brand: ${d.brand} ${(d.brand_conf*100).toFixed(1)}%`);
        else if (d.brand) parts.push(`Brand: ${d.brand}`);
        if (typeof d.grade_conf === 'number' && d.grade) parts.push(`Grade: ${d.grade} ${(d.grade_conf*100).toFixed(1)}%`);
        else if (d.grade) parts.push(`Grade: ${d.grade}`);
        if (parts.length && suggestBox) suggestBox.textContent = `Model suggests -> ${parts.join(' | ')}`;
      });
    }
  });

  // helper to fill datalist options
  function setDataList(datalistEl, items) {
    datalistEl.innerHTML = '';
    items.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      datalistEl.appendChild(opt);
    });
  }

  // prime brand/model lists from CSV (no query -> top/most frequent)
  fetch('/api/label-suggestions?field=brand').then(r => r.json()).then(d => setDataList(brandList, d.items || [])).catch(()=>{});
  fetch('/api/label-suggestions?field=model').then(r => r.json()).then(d => setDataList(modelList, d.items || [])).catch(()=>{});

  // dynamic typeahead: fetch filtered suggestions on input
  let brandDebounce, modelDebounce;
  brandInput.addEventListener('input', () => {
    clearTimeout(brandDebounce);
    const q = encodeURIComponent(brandInput.value.trim());
    brandDebounce = setTimeout(() => {
      fetch(`/api/label-suggestions?field=brand&q=${q}`).then(r => r.json()).then(d => setDataList(brandList, d.items || [])).catch(()=>{});
    }, 150);
  });
  modelInput.addEventListener('input', () => {
    clearTimeout(modelDebounce);
    const q = encodeURIComponent(modelInput.value.trim());
    modelDebounce = setTimeout(() => {
      fetch(`/api/label-suggestions?field=model&q=${q}`).then(r => r.json()).then(d => setDataList(modelList, d.items || [])).catch(()=>{});
    }, 150);
  });

  document.getElementById('save-label').addEventListener('click', () => {
    const payload = {
      ball_id: SESSION_ID,
      brand: brandInput.value,
      model: modelInput.value,
      grade: selectedGrade,
      color: selectedColor
    };
    fetch('/api/label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(r => {
      if (r.ok) {
        alert('Saved');
      } else {
        alert('Save failed');
      }
    });
  });
});

