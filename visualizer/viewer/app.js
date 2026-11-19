// Frame profiles viewer (starter). Reads /api/bundles and /api/bundles/<id> from visualizer/serve.py.

const FAMILIES = ["attention", "matmul", "normalization", "copy / convert", "quantize", "elementwise", "other"];
const FAMILY_VAR = {
  "attention": "--f-attention", "matmul": "--f-matmul", "normalization": "--f-normalization",
  "copy / convert": "--f-copy", "quantize": "--f-quantize", "elementwise": "--f-elementwise",
  "other": "--f-other",
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const familyColor = (family) => css(FAMILY_VAR[family] || "--f-other");
const $ = (selector) => document.querySelector(selector);
const fmt = (x, digits = 0) => (x == null ? "–" : x.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }));
const median = (xs) => { const s = xs.filter((x) => x != null).sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : null; };
const el = (tag, props = {}, children = []) => { const e = Object.assign(document.createElement(tag), props); e.append(...children); return e; };

let bundle = null;
let bundleId = null;
let view = null; // {frame, t0, t1} in µs

async function loadLibrary() {
  const list = await (await fetch("/api/bundles")).json();
  const ul = $("#bundle-list");
  ul.replaceChildren(...list.map((b) => {
    const li = el("li", {}, [
      el("div", { className: "name", textContent: b.meta.name || b.id }),
      el("div", { className: "line", textContent: `${b.id.slice(0, 15)} · ${b.meta.tenants || 1} tenant(s) · ${b.meta.sharing || "timeslice"}` }),
      el("div", { className: "line", textContent: `frame ${fmt(b.client_latency_ms)} ms · GPU ${fmt(b.gpu_span_ms)} ms · idle ${fmt(b.idle_ms, 1)} ms${b.notes ? " · 📝" : ""}` }),
    ]);
    li.onclick = () => select(b.id, li);
    return li;
  }));
  if (!list.length) ul.replaceChildren(el("li", { className: "line", textContent: "No bundles yet: run python -m visualizer.capture" }));
  return list;
}

async function select(id, li) {
  document.querySelectorAll("#bundle-list li").forEach((x) => x.classList.toggle("selected", x === li));
  bundle = await (await fetch(`/api/bundles/${id}`)).json();
  bundleId = id;
  history.replaceState(null, "", `#${id}`); // a link to this profile
  $("#empty").hidden = true;
  $("#bundle").hidden = false;
  renderHeader();
  renderTiles();
  renderBreakdown();
  const frames = bundle.frames.map((f) => f.index);
  $("#frame-select").replaceChildren(...frames.map((i) => el("option", { value: i, textContent: i })));
  $("#frame-select").value = frames.length > 1 ? 1 : 0;
  resetView();
  renderKernels();
  const summary = (await (await fetch("/api/bundles")).json()).find((b) => b.id === id);
  $("#notes").value = summary ? summary.notes : "";
  $("#notes-status").textContent = "";
}

function steadyFrames() { return bundle.frames.length > 1 ? bundle.frames.slice(1) : bundle.frames; }

function renderHeader() {
  const m = bundle.meta;
  $("#title").textContent = m.name || bundleId;
  const extra = (m.server_args || []).join(" ");
  $("#subtitle").textContent = `${bundleId} · ${bundle.gpu.name}, ${bundle.gpu.sms} SMs · ${m.tenants || 1} tenant(s), ${m.load || "none"} neighbours, ${m.sharing || "timeslice"}${extra ? " · " + extra : ""}${m.notes ? " · " + m.notes : ""}`;
  $("#warnings").replaceChildren(...(bundle.warnings || []).map((w) => el("li", { textContent: w })));
}

function renderTiles() {
  const f = steadyFrames();
  const tiles = [
    ["frame, client side", median(f.map((x) => x.client_latency_ms)), "ms"],
    ["GPU span", median(f.map((x) => x.gpu_span_ms)), "ms"],
    ["GPU idle inside the frame", median(f.map((x) => x.idle_ms)), "ms", 1],
    ["outside the GPU", median(f.map((x) => x.client_latency_ms - x.gpu_span_ms)), "ms"],
    ["kernels per frame", median(f.map((x) => x.kernels)), ""],
  ];
  $("#tiles").replaceChildren(...tiles.map(([label, value, unit, digits]) =>
    el("div", { className: "tile" }, [el("div", { className: "label", textContent: label }), el("div", { className: "value", textContent: `${fmt(value, digits || 0)} ${unit}` })])));
}

function stackedBar(label, parts) {
  const total = parts.reduce((s, p) => s + p.ms, 0);
  const bar = el("div", { className: "bar" }, parts.filter((p) => p.ms > 0).map((p) => {
    const d = el("div", { title: `${p.label}: ${fmt(p.ms, 1)} ms (${fmt((100 * p.ms) / total, 1)}%)` });
    d.style.flex = String(p.ms);
    d.style.background = p.color;
    return d;
  }));
  const legend = el("div", { className: "legend" }, parts.filter((p) => p.ms > 0).map((p) => {
    const s = el("span", { textContent: `${p.label} ${fmt(p.ms)} ms (${fmt((100 * p.ms) / total)}%)` });
    s.style.setProperty("--swatch", p.color);
    return s;
  }));
  return [el("div", { className: "bar-label", textContent: label }), bar, legend];
}

function renderBreakdown() {
  const f = steadyFrames();
  const avg = (get) => f.reduce((s, x) => s + (get(x) || 0), 0) / f.length;
  const streams = Object.keys(f[0].by_stream_ms);
  const streamColors = ["--f-attention", "--f-matmul", "--f-normalization"];
  const byStream = streams.map((s, i) => ({ label: s, ms: avg((x) => x.by_stream_ms[s]), color: css(streamColors[i % 3]) }));
  byStream.push({ label: "GPU idle", ms: avg((x) => x.idle_ms), color: css("--idle") });
  byStream.push({ label: "outside the GPU", ms: avg((x) => x.client_latency_ms - x.gpu_span_ms), color: css("--f-other") });
  const byFamily = FAMILIES.map((fam) => ({ label: fam, ms: avg((x) => x.by_family_ms[fam]), color: familyColor(fam) }));
  $("#breakdown").replaceChildren(
    ...stackedBar("By stream (streams can overlap, so this can exceed the GPU span)", byStream),
    ...stackedBar("By kernel family (kernel time)", byFamily));
}

function frameKernels(frame) {
  const k = bundle.kernels;
  const rows = [];
  for (let i = 0; i < k.frame.length; i++) {
    if (k.frame[i] !== frame) continue;
    rows.push({ start: k.start_us[i], dur: k.dur_us[i], name: bundle.names[k.name[i]], stream: k.stream[i], blocks: k.blocks[i], bps: k.blocks_per_sm[i] });
  }
  return rows;
}

function resetView() {
  const frame = Number($("#frame-select").value);
  const rows = frameKernels(frame);
  const end = Math.max(...rows.map((r) => r.start + r.dur));
  view = { frame, rows, t0: 0, t1: end };
  drawTimeline();
}

function drawTimeline() {
  const canvas = $("#timeline");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const left = 120, right = 10, top = 10, laneH = 34, axisY = h - 22;
  const streams = Object.keys(bundle.streams).map(Number).sort((a, b) => a - b);
  const x = (t) => left + ((t - view.t0) / (view.t1 - view.t0)) * (w - left - right);
  ctx.font = "12px system-ui"; ctx.textBaseline = "middle";

  // idle "bubbles": times inside the frame when no stream had a kernel running
  const sorted = [...view.rows].sort((a, b) => a.start - b.start);
  let busyUntil = sorted.length ? sorted[0].start : 0;
  ctx.fillStyle = css("--idle"); ctx.globalAlpha = 0.18;
  for (const r of sorted) {
    if (r.start > busyUntil) ctx.fillRect(x(busyUntil), top, Math.max(1, x(r.start) - x(busyUntil)), streams.length * laneH);
    busyUntil = Math.max(busyUntil, r.start + r.dur);
  }
  ctx.globalAlpha = 1;

  streams.forEach((s, lane) => {
    const y = top + lane * laneH;
    ctx.fillStyle = css("--ink");
    ctx.fillText(bundle.streams[s], 6, y + laneH / 2);
    ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(left, y + laneH); ctx.lineTo(w - right, y + laneH); ctx.stroke();
    for (const r of view.rows) {
      if (r.stream !== s || r.start + r.dur < view.t0 || r.start > view.t1) continue;
      ctx.fillStyle = familyColor(r.name.family);
      ctx.fillRect(x(r.start), y + 6, Math.max(1, x(r.start + r.dur) - x(r.start) - 0.5), laneH - 12);
    }
  });

  // time axis in ms
  ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--grid");
  const span = (view.t1 - view.t0) / 1000, step = Math.pow(10, Math.floor(Math.log10(span / 5)));
  const nice = span / step > 10 ? step * 2 : step;
  for (let t = Math.ceil(view.t0 / 1000 / nice) * nice; t <= view.t1 / 1000; t += nice) {
    const px = x(t * 1000);
    ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, axisY - 6); ctx.stroke();
    ctx.fillText(`${fmt(t, nice < 1 ? 2 : 0)} ms`, px - 14, axisY + 6);
  }
  view.geometry = { left, right, top, laneH, streams, w };
  $("#legend").replaceChildren(...[...FAMILIES.map((f) => [f, familyColor(f)]), ["GPU idle (bubble)", css("--idle")]].map(([label, color]) => {
    const s = el("span", { textContent: label }); s.style.setProperty("--swatch", color); return s;
  }));
}

function timeAt(px) { const g = view.geometry; return view.t0 + ((px - g.left) / (g.w - g.left - g.right)) * (view.t1 - view.t0); }

function wireTimeline() {
  const canvas = $("#timeline"), tip = $("#tooltip");
  let dragFrom = null;
  canvas.addEventListener("mousemove", (e) => {
    if (!view) return;
    const px = e.offsetX, py = e.offsetY, g = view.geometry, t = timeAt(px);
    const lane = Math.floor((py - g.top) / g.laneH), stream = g.streams[lane];
    const slack = (2 * (view.t1 - view.t0)) / (g.w - g.left - g.right);
    const hit = view.rows.find((r) => r.stream === stream && t >= r.start - slack && t <= r.start + r.dur + slack);
    if (!hit) { tip.hidden = true; return; }
    const waves = hit.blocks / (bundle.gpu.sms * hit.bps);
    tip.innerHTML = `<b>${hit.name.short}</b> · ${hit.name.family}<br>${fmt(hit.dur / 1000, 3)} ms at ${fmt(hit.start / 1000, 2)} ms<br>${hit.blocks} blocks, ${fmt(waves, 2)} waves (model)`;
    tip.hidden = false;
    tip.style.left = `${Math.min(px + 12, canvas.clientWidth - 260)}px`;
    tip.style.top = `${py + 12}px`;
  });
  canvas.addEventListener("mouseleave", () => { tip.hidden = true; });
  canvas.addEventListener("mousedown", (e) => { dragFrom = timeAt(e.offsetX); });
  canvas.addEventListener("mouseup", (e) => {
    if (dragFrom == null) return;
    const to = timeAt(e.offsetX);
    if (Math.abs(to - dragFrom) > (view.t1 - view.t0) / 200) { view.t0 = Math.min(dragFrom, to); view.t1 = Math.max(dragFrom, to); drawTimeline(); }
    dragFrom = null;
  });
  canvas.addEventListener("dblclick", resetView);
  $("#frame-select").addEventListener("change", resetView);
  window.addEventListener("resize", () => view && drawTimeline());
}

function renderKernels() {
  const frames = steadyFrames().map((f) => f.index);
  const k = bundle.kernels, agg = new Map();
  for (let i = 0; i < k.frame.length; i++) {
    if (!frames.includes(k.frame[i])) continue;
    const key = `${k.name[i]}|${k.stream[i]}`;
    const a = agg.get(key) || { name: bundle.names[k.name[i]], stream: k.stream[i], calls: 0, us: 0, blocks: 0, waves: 0 };
    a.calls += 1; a.us += k.dur_us[i]; a.blocks += k.blocks[i]; a.waves += k.blocks[i] / (bundle.gpu.sms * k.blocks_per_sm[i]);
    agg.set(key, a);
  }
  const frameMs = median(steadyFrames().map((f) => f.gpu_span_ms));
  const rows = [...agg.values()].sort((a, b) => b.us - a.us).slice(0, 20);
  const n = frames.length;
  $("#kernels tbody").replaceChildren(...rows.map((a) => el("tr", {}, [
    el("td", { className: "name", textContent: a.name.short, title: a.name.full }),
    el("td", { textContent: a.name.family }),
    el("td", { textContent: bundle.streams[a.stream] }),
    el("td", { className: "num", textContent: fmt(a.calls / n) }),
    el("td", { className: "num", textContent: fmt(a.us / n / 1000, 1) }),
    el("td", { className: "num", textContent: fmt((100 * a.us) / n / 1000 / frameMs, 1) }),
    el("td", { className: "num", textContent: fmt(a.us / a.calls / 1000, 3) }),
    el("td", { className: "num", textContent: fmt(a.blocks / a.calls) }),
    el("td", { className: "num", textContent: fmt(a.waves / a.calls, 1) }),
  ])));
}

$("#save-notes").addEventListener("click", async () => {
  const res = await fetch(`/api/bundles/${bundleId}/notes`, { method: "PUT", body: $("#notes").value });
  $("#notes-status").textContent = res.ok ? "saved" : "could not save";
  loadLibrary();
});

wireTimeline();
loadLibrary().then((list) => {
  // Open the profile named in the URL (#<bundle id>), so a profile can be linked to.
  const id = decodeURIComponent(location.hash.slice(1));
  const index = list.findIndex((b) => b.id === id);
  if (index >= 0) select(id, document.querySelectorAll("#bundle-list li")[index]);
});
