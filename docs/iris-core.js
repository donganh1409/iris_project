// iris-core.js
// The same pipeline as app.py, ported to JavaScript so it runs in a browser:
//   OpenCV BGR2GRAY  ->  cv2.resize(128x128, INTER_LINEAR)
//   skimage.feature.hog(pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9, block_norm='L2-Hys')
//   PCA from models/pca_model.joblib (exported to model/pca.json by scripts/export_web_model.py)
//   cosine similarity, match above a threshold
// Every function is pure, so the numbers can be checked against Python (see scripts/check_web_port.py).

export const SIZE = 128, CELL = 16, BLOCK = 2, BINS = 9;
export const NCELL = SIZE / CELL;                 // 8 cells per side
export const NBLOCK = NCELL - BLOCK + 1;          // 7 blocks per side
export const HOG_LEN = NBLOCK * NBLOCK * BLOCK * BLOCK * BINS; // 1764

// cv2.cvtColor(BGR2GRAY) uses fixed-point weights 4899, 9617, 1868 (sum 2^14) with rounding.
export function grayFromRGBA(rgba, w, h) {
  const g = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < g.length; i++, p += 4) {
    g[i] = (rgba[p] * 4899 + rgba[p + 1] * 9617 + rgba[p + 2] * 1868 + 8192) >> 14;
  }
  return g;
}

// cv2.resize(..., INTER_LINEAR): half-pixel centres, clamped edges, 11-bit fixed-point weights.
export function resizeLinear(src, sw, sh, dw = SIZE, dh = SIZE) {
  const out = new Uint8Array(dw * dh), fx = sw / dw, fy = sh / dh, ONE = 2048;
  const xs0 = new Int32Array(dw), xs1 = new Int32Array(dw), wx = new Int32Array(dw);
  for (let x = 0; x < dw; x++) {
    let s = (x + 0.5) * fx - 0.5, i = Math.floor(s), f = s - i;
    if (i < 0) { i = 0; f = 0 } if (i >= sw - 1) { i = sw - 1; f = 0 }
    xs0[x] = i; xs1[x] = Math.min(i + 1, sw - 1); wx[x] = Math.round(f * ONE);
  }
  for (let y = 0; y < dh; y++) {
    let s = (y + 0.5) * fy - 0.5, j = Math.floor(s), f = s - j;
    if (j < 0) { j = 0; f = 0 } if (j >= sh - 1) { j = sh - 1; f = 0 }
    const r0 = j * sw, r1 = Math.min(j + 1, sh - 1) * sw, wy = Math.round(f * ONE);
    for (let x = 0; x < dw; x++) {
      const a = src[r0 + xs0[x]] * (ONE - wx[x]) + src[r0 + xs1[x]] * wx[x];
      const b = src[r1 + xs0[x]] * (ONE - wx[x]) + src[r1 + xs1[x]] * wx[x];
      out[y * dw + x] = Math.min(255, (a * (ONE - wy) + b * wy + (1 << 21)) >> 22);
    }
  }
  return out;
}

// skimage.feature.hog with the repository's settings. Returns {feat: Float64Array(1764), cells: Float64Array(8*8*9)}.
export function hog(gray) {
  const N = SIZE, mag = new Float64Array(N * N), ori = new Float64Array(N * N);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const i = r * N + c;
    const gr = (r === 0 || r === N - 1) ? 0 : gray[i + N] - gray[i - N];
    const gc = (c === 0 || c === N - 1) ? 0 : gray[i + 1] - gray[i - 1];
    mag[i] = Math.hypot(gc, gr);
    let a = Math.atan2(gr, gc) * 180 / Math.PI; a = ((a % 180) + 180) % 180; ori[i] = a;
  }
  const per = 180 / BINS, cells = new Float64Array(NCELL * NCELL * BINS);
  for (let cr = 0; cr < NCELL; cr++) for (let cc = 0; cc < NCELL; cc++) {
    for (let y = cr * CELL; y < (cr + 1) * CELL; y++) for (let x = cc * CELL; x < (cc + 1) * CELL; x++) {
      const i = y * N + x, b = Math.min(BINS - 1, Math.floor(ori[i] / per));
      cells[(cr * NCELL + cc) * BINS + b] += mag[i];
    }
  }
  for (let k = 0; k < cells.length; k++) cells[k] /= CELL * CELL;
  const feat = new Float64Array(HOG_LEN), eps = 1e-5, blk = new Float64Array(BLOCK * BLOCK * BINS);
  let o = 0;
  for (let br = 0; br < NBLOCK; br++) for (let bc = 0; bc < NBLOCK; bc++) {
    let n = 0, s = 0;
    for (let dr = 0; dr < BLOCK; dr++) for (let dc = 0; dc < BLOCK; dc++) for (let b = 0; b < BINS; b++) {
      const v = cells[((br + dr) * NCELL + (bc + dc)) * BINS + b]; blk[n++] = v; s += v * v;
    }
    let d = Math.sqrt(s + eps * eps), s2 = 0;
    for (let k = 0; k < n; k++) { blk[k] = Math.min(blk[k] / d, 0.2); s2 += blk[k] * blk[k] }
    d = Math.sqrt(s2 + eps * eps);
    for (let k = 0; k < n; k++) feat[o++] = blk[k] / d;
  }
  return { feat, cells };
}

// sklearn PCA.transform (whiten=False): (x - mean) . components^T
export function pcaTransform(x, model) {
  return model.components.map(comp => { let s = 0; for (let i = 0; i < comp.length; i++) s += (x[i] - model.mean[i]) * comp[i]; return s });
}

export function cosine(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return na && nb ? d / Math.sqrt(na * nb) : 0;
}

// Full pipeline from an RGBA image of any size.
export function features(rgba, w, h, model) {
  const gray = resizeLinear(grayFromRGBA(rgba, w, h), w, h);
  const { feat, cells } = hog(gray);
  return { gray, cells, hog: Array.from(feat), pca: model ? pcaTransform(feat, model) : null };
}

// match() from app.py: best cosine score, above the threshold or "Unknown".
export function match(vec, db, key, threshold) {
  const scores = db.map(p => ({ name: p.name, score: cosine(vec, p[key]) })).sort((a, b) => b.score - a.score);
  const best = scores[0];
  return { scores, best: best && best.score > threshold ? best : null, top: best || null };
}
