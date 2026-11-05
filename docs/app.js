// app.js: the web version of app.py. Verify, enrol and manage identities, all in the browser.
import { features, match, cosine, SIZE, NCELL, BINS } from './iris-core.js';

const $ = id => document.getElementById(id);
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const KEY = 'iris-web-db-v1';
// The three identities saved in data/iris_features.joblib (PCA vectors), and the photo each was enrolled from.
const REPO = [
  { name: 'nguyễn văn a', file: 'A', pca: [-2.550577, -0.436795, 0.0] },
  { name: 'trần tuấn b', file: 'B', pca: [1.801097, -1.370697, -0.0] },
  { name: 'hồ thanh c', file: 'C', pca: [0.74948, 1.807492, -0.0] },
];
const SAMPLES = [
  { id: 'A', label: 'Sample A', src: 'samples/A.jpg' },
  { id: 'B', label: 'Sample B', src: 'samples/B.jpg' },
  { id: 'C', label: 'Sample C', src: 'samples/C.jpg' },
  { id: 'A2', label: 'A, new photo', src: 'samples/A.jpg', variant: true },
];
const LINES = { pca: 0.70, hog: 0.90 };

let model = null, db = [], current = null, mode = 'pca', threshold = LINES.pca, tab = 'verify', stream = null;
const title = s => s.replace(/(^|\s)\S/g, m => m.toUpperCase());

/* ---------- images ---------- */
const loadImg = src => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src });
function rgbaOf(source, w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d', { willReadFrequently: true });
  draw(x, c); return { data: x.getImageData(0, 0, w, h).data, w, h, canvas: c };
}
function fromImage(img, variant) {
  const w = img.naturalWidth, h = img.naturalHeight;
  return rgbaOf(img, w, h, x => {
    if (variant) { // a second photo of the same eye: turned 6 degrees, a little closer and brighter
      x.fillStyle = '#000'; x.fillRect(0, 0, w, h); x.translate(w / 2, h / 2); x.rotate(6 * Math.PI / 180); x.scale(1.06, 1.06);
      x.filter = 'brightness(1.12) contrast(1.04)'; x.drawImage(img, -w / 2, -h / 2);
    } else x.drawImage(img, 0, 0);
  });
}
function setPhoto(px, label) {
  const cv = $('photo'), st = $('stage'), r = st.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); const x = cv.getContext('2d');
  const s = Math.min(cv.width / px.w, cv.height / px.h); x.fillStyle = '#1A1A1C'; x.fillRect(0, 0, cv.width, cv.height);
  x.drawImage(px.canvas, (cv.width - px.w * s) / 2, (cv.height - px.h * s) / 2, px.w * s, px.h * s);
  $('empty').hidden = true;
  const f = features(px.data, px.w, px.h, model);
  current = { ...f, px, label };
  drawPipe(); refreshButtons(); resetResult();
}

/* ---------- database ---------- */
function save() { try { localStorage.setItem(KEY, JSON.stringify(db.map(p => ({ ...p, hog: p.hog.map(v => +v.toFixed(5)) })))) } catch (e) {} }
async function repoPeople() {
  const out = [];
  for (const p of REPO) { const img = await loadImg(`samples/${p.file}.jpg`); const px = fromImage(img); const f = features(px.data, px.w, px.h, model);
    out.push({ name: p.name, pca: p.pca, hog: f.hog, source: 'repo' }) }
  return out;
}
async function loadDb() {
  try { const s = JSON.parse(localStorage.getItem(KEY) || 'null'); if (Array.isArray(s) && s.every(p => p.name && p.hog && p.pca)) { db = s; return } } catch (e) {}
  db = await repoPeople(); save();
}

/* ---------- views ---------- */
function refreshButtons() {
  $('go').disabled = !current || !db.length;
  $('save').disabled = !current || !$('name').value.trim();
  $('count').textContent = db.length;
}
function resetResult() {
  const r = $('result'); r.dataset.state = 'idle';
  $('r-h').textContent = current ? (db.length ? 'Ready to look' : 'No one is enrolled yet') : 'Ready when you are';
  $('r-s').textContent = current ? (db.length ? `${current.label}. Press Verify.` : 'Enrol an eye first, or restore the repository identities.') : 'Pick an eye on the left, then press Verify.';
  $('scores').hidden = true;
}
function renderPeople() {
  const ul = $('people'); ul.innerHTML = '';
  if (!db.length) { const li = document.createElement('li'); li.textContent = 'Nobody is enrolled. Add someone in Enrol.'; ul.appendChild(li) }
  db.forEach((p, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="av" aria-hidden="true"></span><span class="grow"><span class="nm"></span><br><span class="src"></span></span><button class="x" type="button">Remove</button>`;
    li.querySelector('.av').textContent = p.name.trim()[0].toUpperCase();
    li.querySelector('.nm').textContent = title(p.name);
    li.querySelector('.src').textContent = p.source === 'repo' ? 'From data/iris_features.joblib' : 'Enrolled in this browser';
    li.querySelector('.x').addEventListener('click', () => {
      const box = document.createElement('span'); box.className = 'confirm';
      box.innerHTML = '<button class="x" type="button">Delete for good</button><button class="x" type="button" style="color:inherit">Keep</button>';
      const [yes, no] = box.querySelectorAll('button');
      yes.addEventListener('click', () => { db.splice(i, 1); save(); renderPeople(); refreshButtons(); resetResult() });
      no.addEventListener('click', renderPeople);
      li.querySelector('.x').replaceWith(box); yes.focus();
    });
    ul.appendChild(li);
  });
  refreshButtons();
}
function drawPipe() {
  const f = current; if (!f) return;
  const p = $('c-photo').getContext('2d'); const s = Math.min(96 / f.px.w, 96 / f.px.h) * 1;
  const side = Math.min(f.px.w, f.px.h); p.clearRect(0, 0, 96, 96); p.drawImage(f.px.canvas, (f.px.w - side) / 2, (f.px.h - side) / 2, side, side, 0, 0, 96, 96);
  const g = $('c-gray').getContext('2d'), im = g.createImageData(SIZE, SIZE);
  for (let i = 0; i < f.gray.length; i++) { im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = f.gray[i]; im.data[i * 4 + 3] = 255 } g.putImageData(im, 0, 0);
  const h = $('c-hog').getContext('2d'); h.fillStyle = '#141414'; h.fillRect(0, 0, 128, 128);
  let mx = 0; for (const v of f.cells) mx = Math.max(mx, v);
  h.lineCap = 'round';
  for (let r = 0; r < NCELL; r++) for (let c = 0; c < NCELL; c++) for (let b = 0; b < BINS; b++) {
    const v = f.cells[(r * NCELL + c) * BINS + b] / (mx || 1); if (v < .04) continue;
    const a = ((b + .5) * 20 + 90) * Math.PI / 180, L = 7.5 * v, cx = c * 16 + 8, cy = r * 16 + 8;
    h.strokeStyle = `rgba(255,225,90,${.25 + .75 * v})`; h.lineWidth = 1.2;
    h.beginPath(); h.moveTo(cx - Math.cos(a) * L, cy - Math.sin(a) * L); h.lineTo(cx + Math.cos(a) * L, cy + Math.sin(a) * L); h.stroke();
  }
  const n = $('c-nums');
  if (mode === 'pca') { n.innerHTML = f.pca.map((v, i) => `<span>PC${i + 1} ${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}</span>`).join(''); $('nums-cap').textContent = '3 numbers (PCA)' }
  else { const top = f.hog.slice(0, 4).map(v => v.toFixed(3)); n.innerHTML = `<span>1,764 numbers</span>${top.map(v => `<span>${v}</span>`).join('')}<span>…</span>`; $('nums-cap').textContent = 'All 1,764 edge numbers' }
}
function setMode(m) {
  mode = m; threshold = LINES[m]; $('thr').value = threshold; $('thr-v').textContent = threshold.toFixed(2);
  $('model-l').textContent = m === 'pca' ? 'HOG → PCA → cosine' : 'HOG → cosine';
  $('model-note').textContent = m === 'pca'
    ? 'The PCA in models/pca_model.joblib was fitted on three photos, so it can only spread eyes along two directions. Strangers can pass. Switch to edge features to see the difference.'
    : 'All 1,764 edge numbers are compared directly. Scores run higher, so the line moves up to 0.90, as in the portfolio demo.';
  drawPipe(); resetResult();
}

/* ---------- verify ---------- */
function verify() {
  if (!current || !db.length) return;
  const st = $('stage'); st.classList.remove('scanning'); void st.offsetWidth; st.classList.add('scanning');
  const r = $('result'); r.dataset.state = 'busy'; $('r-h').textContent = 'Looking'; $('r-s').textContent = 'Comparing edge directions';
  setTimeout(() => {
    const m = match(mode === 'pca' ? current.pca : current.hog, db, mode, threshold), pay = $('paymode').checked;
    const bars = $('bars'); bars.innerHTML = '';
    m.scores.forEach((s, i) => {
      const li = document.createElement('li'); if (i === 0) li.className = 'best' + (m.best ? ' pass' : '');
      const w = Math.max(0, s.score) * 100;
      li.innerHTML = `<span class="n"></span><span class="t"><i style="width:0"></i><b style="left:${threshold * 100}%"></b></span><span class="v">${s.score.toFixed(2)}</span>`;
      li.querySelector('.n').textContent = title(s.name); bars.appendChild(li);
      requestAnimationFrame(() => { li.querySelector('i').style.width = w + '%' });
    });
    $('line-l').textContent = `line ${threshold.toFixed(2)}`; $('scores').hidden = false;
    if (m.best) {
      r.dataset.state = 'ok';
      $('r-h').textContent = pay ? 'Paid 45.000 ₫ to Cà phê Hạt' : `Identity verified: ${title(m.best.name)}`;
      $('r-s').textContent = `${pay ? `Verified as ${title(m.best.name)}. ` : ''}Similarity ${m.best.score.toFixed(2)}, above the ${threshold.toFixed(2)} line.`;
    } else {
      r.dataset.state = 'no';
      $('r-h').textContent = 'Could not verify identity';
      $('r-s').textContent = `${pay ? 'Nothing was charged. ' : ''}Closest is ${title(m.top.name)} at ${m.top.score.toFixed(2)}, below the ${threshold.toFixed(2)} line.`;
    }
  }, reduce ? 0 : 650);
}

/* ---------- enrol ---------- */
function enrol() {
  const name = $('name').value.trim().toLowerCase(), msg = $('enrol-msg');
  if (!current || !name) return;
  if (db.some(p => p.name === name)) { msg.className = 'note no'; msg.textContent = `“${title(name)}” is already enrolled. Choose another name.`; return }
  db.push({ name, hog: current.hog, pca: current.pca, source: 'local' }); save(); renderPeople();
  msg.className = 'note ok'; msg.textContent = `${title(name)} is enrolled. Switch to Verify and look again.`; $('name').value = ''; refreshButtons();
}

/* ---------- camera ---------- */
async function camera(on) {
  const v = $('video'), st = $('stage'), b = $('cam');
  if (!on) { if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; v.hidden = true; st.classList.remove('live'); b.setAttribute('aria-pressed', 'false'); b.textContent = 'Camera'; $('snap').hidden = true; return }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false });
    v.srcObject = stream; await v.play(); v.hidden = false; st.classList.add('live'); $('empty').hidden = true;
    b.setAttribute('aria-pressed', 'true'); b.textContent = 'Stop camera'; $('snap').hidden = false;
  } catch (e) { $('r-h').textContent = 'The camera is not available'; $('r-s').textContent = 'Allow camera access in your browser, or use a sample or an upload instead.'; $('result').dataset.state = 'no' }
}
function snap() {
  const v = $('video'), st = $('stage').getBoundingClientRect(); if (!v.videoWidth) return;
  const scale = Math.max(st.width / v.videoWidth, st.height / v.videoHeight), side = Math.round(.44 * st.width / scale);
  const sx = Math.round((v.videoWidth - side) / 2), sy = Math.round((v.videoHeight - side) / 2);
  const px = rgbaOf(v, side, side, x => x.drawImage(v, sx, sy, side, side, 0, 0, side, side));
  camera(false); markSample(null); setPhoto(px, 'Your camera photo');
}

/* ---------- wiring ---------- */
function markSample(id) { document.querySelectorAll('.sample').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.id === id))) }
function setTab(t) {
  tab = t; ['verify', 'enrol', 'people'].forEach(k => { const on = k === t; $('t-' + k).setAttribute('aria-selected', on); $('t-' + k).tabIndex = on ? 0 : -1; $('p-' + k).hidden = !on });
  if (t === 'people') renderPeople();
}
async function init() {
  model = await (await fetch('model/pca.json')).json();
  await loadDb();
  const box = $('samples');
  for (const s of SAMPLES) {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'sample'; b.dataset.id = s.id; b.setAttribute('aria-pressed', 'false');
    b.innerHTML = `<img alt="" src="${s.src}"><span>${s.label}</span>`; if (s.variant) b.querySelector('img').style.transform = 'rotate(6deg) scale(1.1)';
    b.addEventListener('click', async () => { camera(false); markSample(s.id); const img = await loadImg(s.src); setPhoto(fromImage(img, s.variant), s.label) });
    box.appendChild(b);
  }
  $('file').addEventListener('change', async e => { const f = e.target.files[0]; if (!f) return; camera(false); markSample(null);
    const url = URL.createObjectURL(f); try { const img = await loadImg(url); setPhoto(fromImage(img), 'Your upload') } catch (err) { $('r-h').textContent = 'That file could not be read'; $('r-s').textContent = 'Try a JPEG or PNG close-up of one eye.' } finally { URL.revokeObjectURL(url); e.target.value = '' } });
  $('cam').addEventListener('click', () => camera(!stream));
  $('snap').addEventListener('click', snap);
  $('go').addEventListener('click', verify);
  $('save').addEventListener('click', enrol);
  $('name').addEventListener('input', refreshButtons);
  $('name').addEventListener('keydown', e => { if (e.key === 'Enter') enrol() });
  $('restore').addEventListener('click', async () => { const repo = await repoPeople(); db = [...repo, ...db.filter(p => p.source !== 'repo')]; save(); renderPeople(); resetResult() });
  document.querySelectorAll('input[name="model"]').forEach(r => r.addEventListener('change', () => setMode(r.value)));
  $('thr').addEventListener('input', e => { threshold = +e.target.value; $('thr-v').textContent = threshold.toFixed(2); if ($('result').dataset.state !== 'idle' && $('result').dataset.state !== 'busy') verify() });
  const tabs = ['verify', 'enrol', 'people'];
  tabs.forEach((k, i) => { const el = $('t-' + k); el.addEventListener('click', () => setTab(k)); el.addEventListener('keydown', e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { const n = tabs[(i + (e.key === 'ArrowRight' ? 1 : 2)) % 3]; setTab(n); $('t-' + n).focus() } }) });
  setMode('pca'); renderPeople(); resetResult();
  // start with sample A so the page is never empty
  const first = document.querySelector('.sample[data-id="A2"]'); if (first) first.click();
  window.__iris = { db: () => db, current: () => current, verify, setMode, cosine };
}
init();
