const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

let catalog = [];
let lastScenario = "B";

function radioValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}

function setScreen(id) {
  $$(".screen").forEach((s) => s.classList.toggle("is-on", s.id === `screen-${id}`));
  $$(".nav-item").forEach((b) => b.classList.toggle("is-active", b.dataset.screen === id));
  history.replaceState(null, "", `#${id}`);
}

async function api(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function polyline(xs, ys, sx, sy) {
  return xs.map((x, i) => `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(" ");
}

function drawSeries(svg, data, pred) {
  const w = 640;
  const h = svg.viewBox.baseVal.height || 220;
  const pad = 36;
  const xs = data.times;
  const l1 = data.coherence;
  const pauli = data.observable && data.observable.length === xs.length ? data.observable : null;
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ys = pauli ? l1.concat(pauli) : l1;
  let ymin = Math.min(...ys);
  let ymax = Math.max(...ys);
  if (ymin === ymax) {
    ymin -= 0.1;
    ymax += 0.1;
  }
  const span = ymax - ymin;
  ymin -= span * 0.06;
  ymax += span * 0.06;
  const sx = (x) => pad + ((x - xmin) / (xmax - xmin || 1)) * (w - 2 * pad);
  const sy = (y) => h - pad - ((y - ymin) / (ymax - ymin || 1)) * (h - 2 * pad);
  const xT = sx(data.t_decoh);
  const y0 = ymin < 0 && ymax > 0
    ? `<line x1="${pad}" y1="${sy(0)}" x2="${w - 10}" y2="${sy(0)}" stroke="var(--color-neutral-300)" stroke-width="1"></line>`
    : "";
  let extra = "";
  if (pred) {
    const xP = sx(pred.t_pred);
    const xO = sx(pred.t_obs);
    extra += `<rect x="${pad}" y="10" width="${Math.max(0, xO - pad)}" height="${h - pad - 10}" fill="var(--color-accent-100)" stroke="var(--color-accent)" stroke-width="1"/>`;
    extra += `<line x1="${xP}" y1="10" x2="${xP}" y2="${h - pad}" stroke="var(--color-accent-900)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
    extra += `<text x="${xP + 6}" y="24" font-size="11" fill="var(--color-accent-900)" font-family="monospace">pred ${pred.t_pred.toFixed(2)}</text>`;
  }
  const name = data.observable_name || "σx";
  const legend = pauli
    ? `<text x="${pad + 8}" y="${h - 12}" font-size="11" fill="var(--color-accent)" font-family="system-ui">L1</text>
       <text x="${pad + 36}" y="${h - 12}" font-size="11" fill="var(--color-neutral-600)" font-family="system-ui">${name}</text>`
    : "";
  const pauliLine = pauli
    ? `<polyline points="${polyline(xs, pauli, sx, sy)}" fill="none" stroke="var(--color-neutral-500)" stroke-width="1.2" stroke-dasharray="4 3"></polyline>`
    : "";
  svg.innerHTML = `
    <line x1="${pad}" y1="10" x2="${pad}" y2="${h - pad}" stroke="var(--color-neutral-400)" stroke-width="1"></line>
    <line x1="${pad}" y1="${h - pad}" x2="${w - 10}" y2="${h - pad}" stroke="var(--color-neutral-400)" stroke-width="1"></line>
    ${y0}
    ${extra}
    ${pauliLine}
    <polyline points="${polyline(xs, l1, sx, sy)}" fill="none" stroke="var(--color-accent)" stroke-width="2"></polyline>
    <line x1="${xT}" y1="10" x2="${xT}" y2="${h - pad}" stroke="var(--color-accent-700)" stroke-width="1.5" stroke-dasharray="3 3"></line>
    <text x="${xT + 6}" y="${pred ? 44 : 24}" font-size="12" fill="var(--color-accent-900)" font-family="monospace">t_decoh = ${data.t_decoh.toFixed(2)}</text>
    ${legend}
  `;
}

function fillSelects(ids) {
  const opts = catalog.map((e) => `<option value="${e.id}">${e.id} · ${e.title}</option>`).join("");
  ids.forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.innerHTML = opts;
    if ([...el.options].some((o) => o.value === "E8")) el.value = "E8";
  });
}

function paperMatch(entry, paper) {
  if (paper === "all") return true;
  return String(entry.paper).includes(paper);
}

function renderRunsTable(paper) {
  const body = $("#runs-table tbody");
  body.innerHTML = catalog
    .filter((e) => paperMatch(e, paper))
    .map((e) => `<tr>
      <td class="mono">${e.id}</td><td>${e.title}</td><td>${e.paper}</td>
      <td class="mono">${e.claim}</td><td>${e.scenarios}</td>
      <td>${e.n_csv} CSV</td><td class="mono muted">${e.module}</td>
    </tr>`)
    .join("");
}

async function loadDashboard() {
  const m = await api("/api/meta");
  $("#dash-cards").innerHTML = m.champions.map((c) => {
    const warn = c.r2 < c.goal;
    return `<div class="card${warn ? " is-warn" : ""}">
      <p class="card-kicker">${c.scenario} · ${c.label}${warn ? " ниже цели" : ""}</p>
      <p class="metric">${c.r2.toFixed(3)}</p>
      <p class="card-body">R² · ${c.model} · <span class="mono">${c.run}</span>
        · ${warn ? "ниже цели" : "цель"} ${c.goal.toFixed(2)}</p>
    </div>`;
  }).join("");
  $("#dash-bars").innerHTML = m.champions.map((c) => {
    const pct = Math.max(0, Math.min(100, c.r2 * 100));
    const fill = c.r2 < c.goal ? "bar-warn" : "bar-ok";
    return `<div class="bar-row"><span class="mono">${c.scenario}</span>
      <div class="bar-track"><div class="${fill}" style="width:${pct.toFixed(1)}%"></div></div>
      <span class="mono">${c.r2.toFixed(3)}</span></div>`;
  }).join("");
  $("#dash-auroc").textContent =
    "AUROC: " + m.champions.map((c) => `${c.scenario} ${c.auroc.toFixed(3)}`).join(" · ");
  $("#dash-claims").innerHTML = m.claims.map((c) =>
    `<div class="claim"><span class="tag ${c.ok ? "tag-accent" : "tag-outline"}">${c.id}</span>${c.text}</div>`
  ).join("");
  $("#dash-paper").textContent = m.paper;
  const gb = m.cache.gb;
  $("#side-foot").innerHTML =
    `<p>csv · ${m.cache.n_csv}</p><p>кэш · ${m.cache.n_files} · ${gb} ГБ</p>`;

  const bt = $("#bt-table tbody");
  bt.innerHTML = m.champions.map((c) => `<tr${c.r2 < c.goal ? ' style="background:var(--color-accent-100)"' : ""}>
    <td>${c.scenario} · ${c.label}</td><td class="mono">${c.model}</td>
    <td>${c.r2.toFixed(3)}</td><td>${c.auroc.toFixed(3)}</td><td>${c.goal.toFixed(2)}</td>
  </tr>`).join("");
}

async function loadCatalog() {
  const data = await api("/api/catalog");
  catalog = data.experiments;
  fillSelects(["#run-exp", "#tr-exp", "#cmp-a", "#cmp-b"]);
  const b = $("#cmp-b");
  if (b && [...b.options].some((o) => o.value === "E14")) b.value = "E14";
  renderRunsTable(radioValue("paper") || "all");
  await loadCsv($("#run-exp").value);
}

async function loadCsv(expId) {
  const data = await api(`/api/csv/${expId}`);
  const thead = $("#csv-table thead");
  const tbody = $("#csv-table tbody");
  if (!data.headers.length) {
    thead.innerHTML = "";
    tbody.innerHTML = `<tr><td>нет CSV</td></tr>`;
    return;
  }
  thead.innerHTML = `<tr>${data.headers.map((h) => `<th>${h}</th>`).join("")}</tr>`;
  tbody.innerHTML = data.rows.map((r) =>
    `<tr>${data.headers.map((h) => `<td>${r[h] ?? ""}</td>`).join("")}</tr>`
  ).join("");
}

function renderJob(el, job) {
  el.textContent =
    `status: ${job.status}\njob: ${job.module} ${(job.args || []).join(" ")}\n` +
    `pid: ${job.pid ?? "—"}  elapsed: ${job.elapsed_min} min\n` +
    `returncode: ${job.returncode ?? "—"}\n\n${job.log || ""}`;
}

async function refreshJob() {
  const job = await api("/api/jobs");
  renderJob($("#job-log"), job);
  renderJob($("#tr-log"), job);
}

async function loadModels() {
  const data = await api("/api/models");
  $("#model-cards").innerHTML = data.slots.map((s) => {
    const m = s.meta;
    const body = m
      ? `<p class="card-title">${s.label}</p>
         <p class="mono muted">R² ${(m.r2 ?? 0).toFixed(4)} · MAE ${(m.mae ?? 0).toFixed(4)} · n=${m.n_samples ?? "—"}</p>
         <span class="tag tag-accent">${m.kind || "на диске"}</span>`
      : `<p class="card-title">${s.label}</p><p class="muted">нет чемпиона в checkpoints/</p>
         <span class="tag tag-outline">пусто</span>`;
    return `<div class="card${s.warn ? " is-warn" : ""}">
      <p class="card-kicker">${s.kicker}</p>${body}
      <div class="btn-row"><button class="btn btn-secondary" data-load-pred="${s.kicker.includes("TFIM") ? "E" : s.kicker.includes("XXZ") ? "D" : s.kicker.includes("σz") ? "C" : "B"}">Открыть в прогнозе</button></div>
    </div>`;
  }).join("");
  $$("[data-load-pred]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const sc = btn.dataset.loadPred;
      const radio = document.querySelector(`input[name="pred-sc"][value="${sc}"]`);
      if (radio) radio.checked = true;
      setScreen("predict");
    });
  });
}

async function runSim() {
  const scenario = radioValue("sim-sc");
  lastScenario = scenario;
  $("#sim-meta").textContent = "running…";
  const data = await api("/api/simulate", {
    method: "POST",
    body: JSON.stringify({ scenario, seed: Number($("#sim-seed").value) }),
  });
  $("#sim-t2").textContent = `T₂ = ${data.t_decoh.toFixed(2)}`;
  $("#sim-meta").textContent =
    `${data.n_qubits}q · ${data.interaction} · ${data.dissipator} · J=${data.J.toFixed(2)} · ω=${data.omega.toFixed(2)}`;
  drawSeries($("#sim-svg"), data);
}

async function runPred(scenario) {
  lastScenario = scenario;
  const data = await api("/api/predict", {
    method: "POST",
    body: JSON.stringify({
      scenario,
      seed: Number($("#pred-seed").value),
      horizon: Number($("#pred-hor").value),
      alarm: Number($("#pred-alarm-thr").value),
    }),
  });
  drawSeries($("#pred-svg"), data, data);
  $("#pred-win-kicker").textContent =
    `Окно · t_obs = ${data.t_obs.toFixed(2)} · ${data.source}`;
  $("#pred-rem").textContent = `T₂ − t = ${data.remaining.toFixed(2)}`;
  const riskLabel = `risk P(≤${data.horizon}): <strong>${data.risk.toFixed(2)}</strong>`;
  $("#pred-break").innerHTML =
    `physics prior (OLS): ${data.phys_rem.toFixed(2)}<br>` +
    `pred T₂: ${data.t_pred.toFixed(2)} · true ${data.t_decoh.toFixed(2)}<br>` +
    riskLabel;
  $("#pred-gauge").style.width = `${Math.max(0, Math.min(100, data.risk * 100))}%`;
  $("#pred-thr-line").style.left = `${data.alarm * 100}%`;
  $("#pred-risk-note").textContent = `${data.risk.toFixed(2)} · порог ${data.alarm.toFixed(2)}`;
  const alarm = $("#pred-alarm");
  if (data.alarmed) {
    alarm.classList.remove("hidden");
    $("#pred-alarm-text").innerHTML =
      `<strong>⚠ RISK THRESHOLD EXCEEDED</strong> — P(decoherence ≤ horizon ${data.horizon}) = ${data.risk.toFixed(2)} при t_obs = ${data.t_obs.toFixed(2)}`;
  } else {
    alarm.classList.add("hidden");
  }
}

async function startJob(expId, mode) {
  const job = await api("/api/jobs", {
    method: "POST",
    body: JSON.stringify({ exp_id: expId, mode }),
  });
  renderJob($("#job-log"), job);
  renderJob($("#tr-log"), job);
}

function bind() {
  $$(".nav-item").forEach((b) => b.addEventListener("click", () => setScreen(b.dataset.screen)));
  $$("#paper-filter input").forEach((el) =>
    el.addEventListener("change", () => renderRunsTable(radioValue("paper")))
  );
  $("#run-exp").addEventListener("change", () => loadCsv($("#run-exp").value));
  $("#btn-replay").addEventListener("click", () => startJob($("#run-exp").value, radioValue("mode")));
  $("#btn-cancel").addEventListener("click", async () => {
    renderJob($("#job-log"), await api("/api/jobs/cancel", { method: "POST", body: "{}" }));
  });
  $("#btn-job-ref").addEventListener("click", refreshJob);
  $("#btn-cmp").addEventListener("click", async () => {
    const data = await api("/api/compare", {
      method: "POST",
      body: JSON.stringify({ a: $("#cmp-a").value, b: $("#cmp-b").value }),
    });
    $("#cmp-out").textContent = data.text;
  });
  $("#btn-sim").addEventListener("click", () => runSim().catch((e) => { $("#sim-meta").textContent = e.message; }));
  $("#btn-to-pred").addEventListener("click", () => {
    const sc = radioValue("sim-sc");
    const radio = document.querySelector(`input[name="pred-sc"][value="${sc}"]`);
    if (radio) radio.checked = true;
    setScreen("predict");
    runPred(sc).catch((e) => { $("#pred-rem").textContent = e.message; });
  });
  $("#btn-tr").addEventListener("click", () => startJob($("#tr-exp").value, radioValue("tr-mode")));
  $("#btn-tr-ref").addEventListener("click", refreshJob);
  $("#btn-pred").addEventListener("click", () => {
    runPred(radioValue("pred-sc")).catch((e) => { $("#pred-rem").textContent = e.message; });
  });
}

async function boot() {
  bind();
  const hash = location.hash.replace("#", "");
  if (hash) setScreen(hash);
  await loadDashboard();
  await loadCatalog();
  await loadModels();
  await refreshJob();
}

boot().catch((err) => {
  console.error(err);
  $("#side-foot").textContent = err.message;
});
