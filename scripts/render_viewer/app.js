"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const stage = $("stage");
const ctx = canvas.getContext("2d", { alpha: false });
const state = { scenes: [], scene: null, a: null, b: null, mode: "flip", showing: "a", auto: null,
  scale: 1, x: 0, y: 0, autoFit: true, divider: .5, error: null, ready: false, generation: 0, errorGeneration: 0 };
const images = new Map();
let drag = null;

async function loadImage(url) {
  if (images.has(url)) return images.get(url);
  const promise = new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = async () => { try { await image.decode(); resolve(image); } catch (error) { reject(error); } };
    image.onerror = () => reject(new Error("Could not load the selected image"));
    image.src = url;
  });
  images.set(url, promise);
  try {
    const loaded = await promise;
    if (images.size > 6) {
      for (const key of images.keys()) {
        if (![state.a?.url, state.b?.url, state.error?.url, url].includes(key)) { images.delete(key); break; }
      }
    }
    return loaded;
  } catch (error) { images.delete(url); throw error; }
}

function status(message) { $("status").textContent = message; }
function activeImage() { return state.showing === "a" ? state.a : state.b; }
function stopAuto() {
  clearInterval(state.auto); state.auto = null;
  $("auto").setAttribute("aria-pressed", "false");
}
function dimensions() { const rect = stage.getBoundingClientRect(); return [rect.width, rect.height]; }
function fit() {
  if (!state.a?.bitmap || !state.b?.bitmap) return;
  const [width, height] = dimensions();
  const imageWidth = Math.max(state.a.width, state.b.width), imageHeight = Math.max(state.a.height, state.b.height);
  state.scale = Math.min(width / imageWidth, height / imageHeight);
  state.x = (width - imageWidth * state.scale) / 2;
  state.y = (height - imageHeight * state.scale) / 2;
  state.autoFit = true; draw();
}
function nativePixels() {
  if (!state.ready) return;
  const [width, height] = dimensions();
  state.scale = 1 / window.devicePixelRatio;
  state.x = (width - state.a.width * state.scale) / 2;
  state.y = (height - state.a.height * state.scale) / 2;
  state.autoFit = false; draw();
}
function layer(bitmap) {
  ctx.drawImage(bitmap, state.x, state.y, bitmap.naturalWidth * state.scale, bitmap.naturalHeight * state.scale);
}
function draw() {
  const [width, height] = dimensions();
  const dpr = window.devicePixelRatio || 1;
  const actualWidth = Math.round(width * dpr), actualHeight = Math.round(height * dpr);
  if (canvas.width !== actualWidth || canvas.height !== actualHeight) { canvas.width = actualWidth; canvas.height = actualHeight; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#101114"; ctx.fillRect(0, 0, width, height);
  ctx.imageSmoothingEnabled = !$("nearest").checked;
  if (!state.ready) return;
  let label;
  if (state.mode === "error") {
    if (state.error?.bitmap) layer(state.error.bitmap);
    else layer(state.a.bitmap);
    label = state.error ? `FLIP mean ${state.error.mean_error.toFixed(4)} · ${state.error.ppd.toFixed(1)} px/degree` : "FLIP map loading…";
  } else if (state.mode === "wipe") {
    layer(state.a.bitmap);
    ctx.save(); ctx.beginPath(); ctx.rect(width * state.divider, 0, width, height); ctx.clip();
    layer(state.b.bitmap); ctx.restore();
    ctx.strokeStyle = "#e6d0a6"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(width * state.divider, 0); ctx.lineTo(width * state.divider, height); ctx.stroke();
    label = `A: ${state.a.label} | B: ${state.b.label}`;
  } else {
    layer(activeImage().bitmap);
    label = `${state.showing.toUpperCase()} · ${activeImage().label}`;
  }
  $("current").textContent = label;
  $("current").className = `current ${state.mode === "flip" ? state.showing : ""}`;
  $("fullscreen-label").textContent = label;
  canvas.dataset.frame = state.mode === "flip" ? state.showing : state.mode;
  canvas.dataset.zoom = (state.scale * dpr).toFixed(4);
}

function fillSelections() {
  const previous = [$("first").value, $("second").value];
  const entries = state.scene.entries.filter(e => e.kind !== "intermediate" || $("intermediates").checked);
  for (const [index, name] of ["first", "second"].entries()) {
    const select = $(name); select.replaceChildren();
    for (const entry of entries) select.add(new Option(entry.label, entry.id));
    select.value = entries.some(e => e.id === previous[index]) ? previous[index] :
      (index === 0 ? entries[0].id : (entries.find(e => e.label === "Brightness + Wasserstein") || entries.find(e => e.kind === "final") || entries[0]).id);
  }
}
async function selectPair(resetView = false) {
  const generation = ++state.generation;
  ++state.errorGeneration; stopAuto(); state.error = null; state.ready = false;
  $("compute").disabled = false;
  $("empty").hidden = false; $("empty").textContent = "Loading images…";
  status("Loading both images before flipping");
  const a = state.scene.entries.find(e => e.id === $("first").value);
  const b = state.scene.entries.find(e => e.id === $("second").value);
  try {
    const [bitmapA, bitmapB] = await Promise.all([loadImage(a.url), loadImage(b.url)]);
    if (generation !== state.generation) return;
    state.a = { ...a, bitmap: bitmapA }; state.b = { ...b, bitmap: bitmapB };
    state.ready = true; state.showing = "a"; $("empty").hidden = true;
    if (resetView || state.autoFit) fit(); else draw();
    status(a.width === b.width && a.height === b.height ? `${a.width} × ${a.height} · same zoom and position` : "Different dimensions · same pixel scale, aligned at top left");
    if (state.mode === "error") await computeError();
  } catch (error) {
    if (generation !== state.generation) return;
    $("empty").textContent = error.message; status(error.message);
  }
}
function flip() {
  if (!state.ready || state.mode !== "flip") return;
  state.showing = state.showing === "a" ? "b" : "a"; draw();
}
function toggleAuto() {
  if (!state.ready) return;
  if (state.auto) { stopAuto(); return; }
  setMode("flip");
  state.auto = setInterval(flip, Number($("interval").value));
  $("auto").setAttribute("aria-pressed", "true");
}
async function setMode(mode) {
  stopAuto(); state.mode = mode;
  for (const button of document.querySelectorAll("[data-mode]")) button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  $("flip-controls").hidden = mode !== "flip";
  $("wipe-controls").hidden = mode !== "wipe";
  $("error-controls").hidden = mode !== "error";
  draw();
  if (mode === "error") await computeError();
}
async function computeError() {
  if (!state.ready) return;
  const generation = ++state.errorGeneration;
  $("compute").disabled = true; status("Computing official FLIP map for A against B…");
  try {
    const response = await fetch("/api/flip", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a: state.a.id, b: state.b.id, ppd: Number($("ppd").value) }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    const bitmap = await loadImage(result.url);
    if (generation !== state.errorGeneration || state.mode !== "error") return;
    state.error = { ...result, bitmap }; draw();
    status("Nominal viewing geometry · zoom does not change map parameters");
  } catch (error) { if (generation === state.errorGeneration) status(error.message); }
  finally { if (generation === state.errorGeneration) $("compute").disabled = false; }
}
async function fullscreen() {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await stage.requestFullscreen(); }
  catch { status("Use your browser’s fullscreen control"); }
}

$("scene").addEventListener("change", () => {
  state.scene = state.scenes.find(s => s.id === $("scene").value);
  $("ppd").value = state.scene.ppd.toFixed(1); fillSelections(); selectPair(true);
});
$("first").addEventListener("change", () => selectPair());
$("second").addEventListener("change", () => selectPair());
$("intermediates").addEventListener("change", () => { fillSelections(); selectPair(); });
$("swap").addEventListener("click", () => { const a = $("first").value; $("first").value = $("second").value; $("second").value = a; selectPair(); });
$("flip").addEventListener("click", () => { stopAuto(); flip(); });
$("auto").addEventListener("click", toggleAuto);
$("interval").addEventListener("input", () => { $("interval-value").value = `${$("interval").value} ms`; if (state.auto) { stopAuto(); toggleAuto(); } });
$("divider").addEventListener("input", () => { state.divider = Number($("divider").value) / 100; draw(); });
for (const button of document.querySelectorAll("[data-mode]")) button.addEventListener("click", () => setMode(button.dataset.mode));
$("compute").addEventListener("click", computeError);
$("ppd").addEventListener("change", () => { if (state.mode === "error") computeError(); });
$("fit").addEventListener("click", fit); $("native").addEventListener("click", nativePixels);
$("nearest").addEventListener("change", draw); $("fullscreen").addEventListener("click", fullscreen);
new ResizeObserver(() => { if (state.autoFit && state.ready) fit(); else draw(); }).observe(stage);
canvas.addEventListener("wheel", (event) => {
  event.preventDefault(); if (!state.ready) return;
  const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
  const previous = state.scale, next = Math.max(.01, Math.min(16 / window.devicePixelRatio, previous * Math.exp(-event.deltaY * .0015)));
  state.x = x - (x - state.x) * next / previous; state.y = y - (y - state.y) * next / previous;
  state.scale = next; state.autoFit = false; draw();
}, { passive: false });
canvas.addEventListener("pointerdown", (event) => {
  if (!state.ready) return;
  canvas.focus(); canvas.setPointerCapture(event.pointerId);
  const rect = canvas.getBoundingClientRect();
  drag = { x: event.clientX, y: event.clientY, startX: state.x, startY: state.y,
    wipe: state.mode === "wipe" && Math.abs(event.clientX - rect.left - rect.width * state.divider) < 15 };
});
canvas.addEventListener("pointermove", (event) => {
  if (!drag) return;
  if (drag.wipe) {
    const rect = canvas.getBoundingClientRect();
    state.divider = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)); $("divider").value = String(state.divider * 100);
  } else {
    state.x = drag.startX + event.clientX - drag.x; state.y = drag.startY + event.clientY - drag.y; state.autoFit = false;
  }
  draw();
});
canvas.addEventListener("pointerup", () => { drag = null; });
canvas.addEventListener("pointercancel", () => { drag = null; });
document.addEventListener("visibilitychange", () => { if (document.hidden) stopAuto(); });
document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
  if (event.target.tagName === "BUTTON" && ["Space", "Enter"].includes(event.code)) return;
  if (event.code === "Space") { event.preventDefault(); stopAuto(); setMode("flip"); flip(); }
  else if (event.code === "Enter") { event.preventDefault(); toggleAuto(); }
  else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault(); setMode("flip"); state.showing = event.key === "ArrowLeft" ? "a" : "b"; draw();
  } else if (event.key.toLowerCase() === "f") fullscreen();
  else if (event.key === "0") fit(); else if (event.key === "1") nativePixels();
  else if (event.key.toLowerCase() === "w") setMode("wipe"); else if (event.key.toLowerCase() === "e") setMode("error");
});

(async () => {
  try {
    const response = await fetch("/api/catalog");
    if (!response.ok) throw new Error("Could not read the render catalog");
    state.scenes = (await response.json()).scenes;
    if (!state.scenes.length) throw new Error("No renders found");
    for (const scene of state.scenes) $("scene").add(new Option(scene.label, scene.id));
    state.scene = state.scenes[0]; $("ppd").value = state.scene.ppd.toFixed(1);
    fillSelections(); await selectPair(true);
  } catch (error) { $("empty").textContent = error.message; status(error.message); }
})();
