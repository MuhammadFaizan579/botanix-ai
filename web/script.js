// Flower classifier: runs model.onnx in the browser with ONNX Runtime Web.

const MODEL_URL = "model.onnx";
const CLASSES_URL = "class_names.json";
const SIZE = 224;                       // model input size
const MEAN = [0.485, 0.456, 0.406];     // ImageNet mean (same as training)
const STD  = [0.229, 0.224, 0.225];     // ImageNet std (same as training)
const LOW_CONFIDENCE = 0.5;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;            // works without special server headers

const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const demo = document.getElementById("demo");
const resultView = document.getElementById("resultView");
const preview = document.getElementById("preview");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

let session = null;
let classNames = null;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

async function init() {
  try {
    const [s, names] = await Promise.all([
      ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] }),
      fetch(CLASSES_URL).then(r => {
        if (!r.ok) throw new Error("class_names.json not found");
        return r.json();
      }),
    ]);
    session = s;
    classNames = names;
    setStatus("Model ready. Choose a flower photo.");
  } catch (err) {
    console.error(err);
    setStatus("Could not load the model. Make sure model.onnx and class_names.json are in the same folder as index.html.", true);
  }
}

// Same as training: Resize(256) -> CenterCrop(224) -> normalise -> CHW tensor
function imageToTensor(img) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  const w = img.naturalWidth, h = img.naturalHeight;
  const side = Math.min(w, h) * (SIZE / 256);        // centre-crop region in source pixels
  ctx.drawImage(img, (w - side) / 2, (h - side) / 2, side, side, 0, 0, SIZE, SIZE);

  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      data[c * plane + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function showResults(probs) {
  const top = probs
    .map((p, i) => ({ p, name: classNames[i] }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3);

  resultsEl.replaceChildren();
  for (const t of top) {
    const li = document.createElement("li");
    const row = document.createElement("div");
    row.className = "row";
    const name = document.createElement("span");
    name.textContent = t.name;
    const pct = document.createElement("span");
    pct.textContent = (t.p * 100).toFixed(1) + "%";
    row.append(name, pct);
    const bar = document.createElement("div");
    bar.className = "meter";
    const fill = document.createElement("i");
    bar.append(fill);
    li.append(row, bar);
    resultsEl.append(li);
    requestAnimationFrame(() => { fill.style.width = (t.p * 100).toFixed(1) + "%"; });
  }

  if (top[0].p < LOW_CONFIDENCE) {
    setStatus("Low confidence. This may not be one of the 102 species, or the photo may be unclear.");
  } else {
    setStatus("");
  }
}

async function classify(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Please choose an image file (JPG, PNG or WebP).", true);
    return;
  }
  if (!session) {
    setStatus("The model is still loading. Try again in a moment.", true);
    return;
  }

  const url = URL.createObjectURL(file);
  preview.src = url;
  drop.hidden = true;
  resultView.hidden = false;
  resultsEl.replaceChildren();
  setStatus("Analysing…");

  try {
    await preview.decode();
    const input = imageToTensor(preview);
    const output = await session.run({ input });
    showResults(softmax(output.output.data));
  } catch (err) {
    console.error(err);
    setStatus("Something went wrong while reading this image. Try a different photo.", true);
  }
}

fileInput.addEventListener("change", () => classify(fileInput.files[0]));

["dragenter", "dragover"].forEach(ev =>
  demo.addEventListener(ev, e => { e.preventDefault(); demo.classList.add("over"); }));
["dragleave", "drop"].forEach(ev =>
  demo.addEventListener(ev, e => { e.preventDefault(); demo.classList.remove("over"); }));
demo.addEventListener("drop", e => classify(e.dataTransfer.files[0]));

init();

// ---- Project links: status badges, copy buttons, active nav ----
const cards = [...document.querySelectorAll("[data-link]")];
function resolved(card) {
  const a = card.querySelector("a.open");
  return { url: a.href, pending: a.getAttribute("href").includes("REPLACE_WITH") };
}
function flash(btn, text) {
  const old = btn.dataset.label || btn.textContent;
  btn.dataset.label = old;
  btn.textContent = text;
  clearTimeout(btn._t);
  btn._t = setTimeout(() => { btn.textContent = old; }, 1600);
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; }
  catch {
    const t = document.createElement("textarea");
    t.value = text; document.body.append(t); t.select();
    const ok = document.execCommand("copy"); t.remove(); return ok;
  }
}
cards.forEach(card => {
  const info = resolved(card);
  const badge = card.querySelector(".badge");
  const open = card.querySelector("a.open");
  const copy = card.querySelector("button.copy");
  badge.textContent = info.pending ? "Link pending" : "Ready";
  badge.classList.toggle("pending", info.pending);
  open.addEventListener("click", e => { if (info.pending) { e.preventDefault(); flash(open, "Not added yet"); } });
  copy.addEventListener("click", async () => {
    if (info.pending) return flash(copy, "Not added yet");
    flash(copy, (await copyText(info.url)) ? "Copied ✓" : "Copy failed");
  });
});
document.getElementById("copyAll").addEventListener("click", async e => {
  const btn = e.currentTarget;
  const ready = cards.map(c => ({ t: c.querySelector("h3").textContent, ...resolved(c) })).filter(x => !x.pending);
  if (!ready.length) return flash(btn, "No links added yet");
  const ok = await copyText(ready.map(x => x.t + ": " + x.url).join("\n"));
  flash(btn, ok ? "Copied " + ready.length + " links ✓" : "Copy failed");
});
const navLinks = [...document.querySelectorAll("#nav a")];
const io = new IntersectionObserver(entries => entries.forEach(en => {
  if (en.isIntersecting) navLinks.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + en.target.id));
}), { rootMargin: "-40% 0px -55% 0px" });
navLinks.forEach(a => { const t = document.querySelector(a.getAttribute("href")); if (t) io.observe(t); });
