// Screen 8, "Конструктор": assemble a recipe from blocks, preview, save, run.
// Loaded after workbench.js, whose helpers ($, $$, api, renderJob) it uses.

const CTOR_STAGES = ["system", "channels", "initial", "evolve", "observe", "target"];
const CTOR_COLORS = ["#5980a6", "#b0623f", "#4f8a5b", "#8a5aa0", "#9a8a2f"];
const CTOR_TIME_PARAMS = [
  { name: "t_max", kind: "float", required: true, nullable: false, default: 20, injected: null },
  { name: "dt", kind: "float", required: true, nullable: false, default: 0.1, injected: null },
];

let ctorCatalog = null;
let ctorState = null;
let ctorPreview = null;
let ctorPoll = null;

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- catalog access ----

function ctorBlocks(stage) {
  return ctorCatalog.stages[stage];
}

function ctorBlock(fn) {
  for (const stage of Object.values(ctorCatalog.stages)) {
    const found = stage.find((b) => b.fn === fn);
    if (found) return found;
  }
  throw new Error(`блок ${fn} не найден в каталоге`);
}

function ctorOutput() {
  // Before the state exists (ctorSkeleton), the default evolve block applies.
  const fn = ctorState ? ctorState.evolve.fn : "evolve.master_equation";
  return ctorCatalog.evolve_output[fn];
}

function ctorCandidates(stage, role) {
  const output = ctorOutput();
  return ctorBlocks(stage).filter((b) => {
    if (role && b.role !== role) return false;
    if (stage === "observe" && b.role === "producer") return b.needs === output;
    if (stage === "target") return b.needs === output;
    return true;
  });
}

// ---- state ----

function ctorFormatValue(def, value) {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

function ctorNewParam(def, value, hasValue) {
  let v = hasValue ? value : (def.required ? null : def.default);
  if (def.choices && v === null) v = def.choices[0];
  return {
    raw: ctorFormatValue(def, v),
    checked: v === true,
    vary: false,
    lo: "",
    hi: "",
  };
}

function ctorShownParams(blockDef) {
  return blockDef.params.filter((p) => !p.injected);
}

function ctorNewItem(fn) {
  const def = ctorBlock(fn);
  const p = {};
  ctorShownParams(def).forEach((d) => { p[d.name] = ctorNewParam(d, null, false); });
  return { fn, p };
}

function ctorNewTime() {
  const p = {};
  CTOR_TIME_PARAMS.forEach((d) => { p[d.name] = ctorNewParam(d, d.default, true); });
  return { p };
}

function ctorSkeleton() {
  const first = (stage, role) => ctorCandidates(stage, role)[0].fn;
  return {
    system: ctorNewItem(first("system")),
    channels: [ctorNewItem(first("channels", "producer"))],
    initial: ctorNewItem(first("initial")),
    evolve: ctorNewItem("evolve.master_equation"),
    observe: [ctorNewItem(first("observe", "producer"))],
    target: null,
    time: ctorNewTime(),
    seeds: "spawn",
    legacyOffset: 0,
  };
}

function ctorItem(stage, idx) {
  if (stage === "channels" || stage === "observe") return ctorState[stage][idx];
  return ctorState[stage];
}

function ctorDefs(stage, item) {
  return stage === "time" ? CTOR_TIME_PARAMS : ctorShownParams(ctorBlock(item.fn));
}

function ctorCanVary(def) {
  return (def.kind === "float" || def.kind === "int") && !def.choices && !def.injected;
}

// Every item of the state with its stage, index and parameter definitions.
function ctorEntries() {
  const out = [];
  const s = ctorState;
  out.push(["system", 0, s.system]);
  s.channels.forEach((it, i) => out.push(["channels", i, it]));
  out.push(["initial", 0, s.initial]);
  out.push(["evolve", 0, s.evolve]);
  s.observe.forEach((it, i) => out.push(["observe", i, it]));
  if (s.target) out.push(["target", 0, s.target]);
  out.push(["time", 0, s.time]);
  return out;
}

// Name of the sampled Param of every varied parameter, keyed "stage:idx:param".
// A name used by several parameters is prefixed with the block function.
function ctorParamNames() {
  const varied = [];
  ctorEntries().forEach(([stage, idx, item]) => {
    ctorDefs(stage, item).forEach((def) => {
      if (ctorCanVary(def) && item.p[def.name].vary) {
        const owner = stage === "time" ? "time" : ctorBlock(item.fn).name;
        varied.push({ key: `${stage}:${idx}:${def.name}`, param: def.name, owner });
      }
    });
  });
  const count = {};
  varied.forEach((v) => { count[v.param] = (count[v.param] || 0) + 1; });
  const used = new Set();
  const names = {};
  varied.forEach((v) => {
    const base = count[v.param] > 1 ? `${v.owner}_${v.param}` : v.param;
    let name = base;
    for (let k = 2; used.has(name); k += 1) name = `${base}_${k}`;
    used.add(name);
    names[v.key] = name;
  });
  return names;
}

// ---- state -> recipe ----

function ctorParseTuple(text) {
  return text.split(",").map((t) => t.trim()).filter((t) => t !== "").map((t) =>
    (Number.isFinite(Number(t)) ? Number(t) : t));
}

function ctorValue(def, st) {
  if (def.kind === "bool") return st.checked;
  const raw = st.raw.trim();
  if (raw === "") return undefined;
  if (def.kind === "tuple") return ctorParseTuple(raw);
  if (def.kind === "str") return raw;
  return Number(raw);
}

function ctorBuildBlock(stage, idx, item, names, ranges) {
  const params = {};
  ctorDefs(stage, item).forEach((def) => {
    const st = item.p[def.name];
    if (ctorCanVary(def) && st.vary) {
      const name = names[`${stage}:${idx}:${def.name}`];
      params[def.name] = { param: name };
      ranges[name] = [Number(st.lo), Number(st.hi)];
      return;
    }
    const value = ctorValue(def, st);
    if (value !== undefined) params[def.name] = value;
    else if (def.nullable && def.required) params[def.name] = null;
  });
  return { fn: item.fn, params };
}

function ctorBuildRecipe() {
  const s = ctorState;
  const names = ctorParamNames();
  const ranges = {};
  const b = (stage, idx, item) => ctorBuildBlock(stage, idx, item, names, ranges);
  const recipe = {
    system: b("system", 0, s.system),
    channels: s.channels.map((it, i) => b("channels", i, it)),
    initial: b("initial", 0, s.initial),
    evolve: b("evolve", 0, s.evolve),
    observe: s.observe.map((it, i) => b("observe", i, it)),
    target: s.target ? b("target", 0, s.target) : null,
  };
  CTOR_TIME_PARAMS.forEach((def) => {
    const st = s.time.p[def.name];
    if (st.vary) {
      const name = names[`time:0:${def.name}`];
      recipe[def.name] = { param: name };
      ranges[name] = [Number(st.lo), Number(st.hi)];
    } else {
      recipe[def.name] = Number(st.raw);
    }
  });
  if (Object.keys(ranges).length) {
    recipe.sampler = { fn: "samplers.uniform", params: ranges };
  }
  if (s.seeds !== "spawn") {
    recipe.seeds = s.seeds;
    recipe.legacy_rng_offset = s.legacyOffset;
  }
  return recipe;
}

// ---- recipe -> state ----

function ctorFromRecipe(recipe) {
  const ranges = {};
  if (recipe.sampler) {
    if (recipe.sampler.fn !== "samplers.uniform") {
      throw new Error(`форма поддерживает только samplers.uniform, в рецепте ${recipe.sampler.fn}`);
    }
    Object.assign(ranges, recipe.sampler.params || {});
  }
  const used = new Set();
  const load = (where, def, st, value) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      if (!(value.param in ranges)) {
        throw new Error(`${where}: параметр ${value.param} не задан в uniform-сэмплере`);
      }
      if (!ctorCanVary(def)) throw new Error(`${where}: ${def.name} нельзя варьировать`);
      used.add(value.param);
      st.vary = true;
      st.lo = String(ranges[value.param][0]);
      st.hi = String(ranges[value.param][1]);
      return;
    }
    const filled = ctorNewParam(def, value, true);
    Object.assign(st, filled);
  };
  const item = (where, stage, data) => {
    const def = ctorBlock(data.fn);
    if (def.stage !== stage) throw new Error(`${where}: ${data.fn} не блок стадии ${stage}`);
    const it = ctorNewItem(data.fn);
    const given = data.params || {};
    Object.keys(given).forEach((k) => {
      if (!(k in it.p)) throw new Error(`${where}: у ${data.fn} нет параметра ${k} в форме`);
    });
    ctorShownParams(def).forEach((d) => {
      if (d.name in given) load(`${where}.${d.name}`, d, it.p[d.name], given[d.name]);
      else if (d.required) it.p[d.name] = ctorNewParam(d, null, false);
    });
    return it;
  };
  const time = ctorNewTime();
  CTOR_TIME_PARAMS.forEach((d) => {
    if (d.name in recipe) load(d.name, d, time.p[d.name], recipe[d.name]);
  });
  const state = {
    system: item("system", "system", recipe.system),
    channels: recipe.channels.map((b, i) => item(`channels[${i}]`, "channels", b)),
    initial: item("initial", "initial", recipe.initial),
    evolve: item("evolve", "evolve", recipe.evolve || { fn: "evolve.master_equation" }),
    observe: recipe.observe.map((b, i) => item(`observe[${i}]`, "observe", b)),
    target: recipe.target ? item("target", "target", recipe.target) : null,
    time,
    seeds: recipe.seeds || "spawn",
    legacyOffset: recipe.legacy_rng_offset || 0,
  };
  Object.keys(ranges).forEach((name) => {
    if (!used.has(name)) throw new Error(`сэмплер задаёт ${name}, но рецепт его не использует`);
  });
  return state;
}

// ---- rendering of the pipeline ----

function ctorParamRow(stage, idx, item, def) {
  const st = item.p[def.name];
  const at = `data-stage="${stage}" data-idx="${idx}" data-param="${esc(def.name)}"`;
  const star = def.required && !def.nullable ? ' <span class="mono" title="обязательный">*</span>' : "";
  const label = `<label>${esc(def.name)}${star}</label>`;
  let control;
  if (def.choices) {
    control = `<select class="input" ${at} data-field="raw">${def.choices.map((c) =>
      `<option value="${esc(c)}"${String(c) === st.raw ? " selected" : ""}>${esc(c)}</option>`
    ).join("")}</select>`;
  } else if (def.kind === "bool") {
    control = `<input type="checkbox" ${at} data-field="checked"${st.checked ? " checked" : ""}>`;
  } else if (ctorCanVary(def) && st.vary) {
    control = `<span class="ctor-range">
      <input class="input" type="number" step="any" ${at} data-field="lo" value="${esc(st.lo)}" aria-label="от">
      <input class="input" type="number" step="any" ${at} data-field="hi" value="${esc(st.hi)}" aria-label="до">
    </span>
    <span class="muted mono" data-pname="${stage}:${idx}:${esc(def.name)}"></span>`;
  } else if (def.kind === "float" || def.kind === "int") {
    const step = def.kind === "int" ? "1" : "any";
    const hint = def.nullable ? ' placeholder="пусто = null"' : "";
    control = `<input class="input" type="number" step="${step}" ${at} data-field="raw" value="${esc(st.raw)}"${hint}>`;
  } else {
    const hint = def.kind === "tuple" ? ' placeholder="через запятую"' : "";
    control = `<input class="input" type="text" ${at} data-field="raw" value="${esc(st.raw)}"${hint}>`;
  }
  const vary = ctorCanVary(def)
    ? `<label class="ctor-vary"><input type="checkbox" ${at} data-field="vary"${st.vary ? " checked" : ""}> варьировать</label>`
    : "";
  return `<div class="field ctor-param">${label}<div class="ctor-control">${control}${vary}</div></div>`;
}

function ctorInjectedNotes(def) {
  return def.params.filter((p) => p.injected).map((p) =>
    `<p class="muted mono ctor-note">${esc(p.name)} — из блока «Время»</p>`).join("");
}

function ctorBlockSelect(stage, idx, item, options) {
  const at = `data-stage="${stage}" data-idx="${idx}" data-field="fn"`;
  const none = stage === "target"
    ? `<option value=""${item ? "" : " selected"}>нет</option>` : "";
  const opts = options.map((b) =>
    `<option value="${esc(b.fn)}" title="${esc(b.doc)}"${item && item.fn === b.fn ? " selected" : ""}>${esc(b.name)}</option>`
  ).join("");
  return `<select class="input" ${at}>${none}${opts}</select>`;
}

function ctorItemBody(stage, idx, item) {
  const def = ctorBlock(item.fn);
  const rows = ctorShownParams(def).map((d) => ctorParamRow(stage, idx, item, d)).join("");
  return `<p class="muted ctor-doc">${esc(def.doc)}</p>${ctorInjectedNotes(def)}${rows}`;
}

function ctorListItem(stage, idx, item, count) {
  const def = ctorBlock(item.fn);
  const role = def.role;
  const list = ctorState[stage];
  const trial = (next) => next.length === 0 || ctorBlock(next[0].fn).role === "producer";
  const swap = (a, b) => {
    const copy = list.slice();
    [copy[a], copy[b]] = [copy[b], copy[a]];
    return copy;
  };
  const canUp = idx > 0 && trial(swap(idx, idx - 1));
  const canDown = idx < count - 1 && trial(swap(idx, idx + 1));
  const canRemove = trial(list.filter((_, i) => i !== idx));
  const at = `data-stage="${stage}" data-idx="${idx}"`;
  const tagClass = role === "producer" ? "tag-accent" : "tag-outline";
  const tagText = role === "producer" ? "источник" : (stage === "observe" ? "преобразование" : "модификатор");
  return `<div class="ctor-item">
    <div class="ctor-item-head">
      <span class="tag ${tagClass}">${tagText}</span>
      ${ctorBlockSelect(stage, idx, item, ctorCandidates(stage, role))}
      <button class="btn btn-ghost" ${at} data-action="up"${canUp ? "" : " disabled"} title="выше">↑</button>
      <button class="btn btn-ghost" ${at} data-action="down"${canDown ? "" : " disabled"} title="ниже">↓</button>
      <button class="btn btn-ghost" ${at} data-action="remove"${canRemove ? "" : " disabled"} title="удалить">✕</button>
    </div>
    ${ctorItemBody(stage, idx, item)}
  </div>`;
}

function ctorCard(title, body) {
  return `<div class="card ctor-card"><p class="card-kicker">${title}</p>${body}</div>`;
}

function ctorRenderPipeline() {
  const s = ctorState;
  const single = (stage, title, options) =>
    ctorCard(title, ctorBlockSelect(stage, 0, s[stage], options) + ctorItemBody(stage, 0, s[stage]));
  const list = (stage, title, addButtons) =>
    ctorCard(title,
      s[stage].map((it, i) => ctorListItem(stage, i, it, s[stage].length)).join("") +
      `<div class="btn-row">${addButtons}</div>`);
  const channelButtons =
    `<button class="btn btn-secondary" data-stage="channels" data-action="add-producer">+ канал</button>` +
    `<button class="btn btn-secondary" data-stage="channels" data-action="add-modifier"` +
    `${s.channels.length === 0 ? " disabled" : ""}>+ модификатор</button>`;
  const observeButtons =
    `<button class="btn btn-secondary" data-stage="observe" data-action="add-producer">+ наблюдение</button>` +
    `<button class="btn btn-secondary" data-stage="observe" data-action="add-modifier"` +
    `${s.observe.length === 0 ? " disabled" : ""}>+ преобразование</button>`;
  const targetBody = ctorBlockSelect("target", 0, s.target, ctorCandidates("target")) +
    (s.target ? ctorItemBody("target", 0, s.target)
      : '<p class="muted ctor-doc">Без цели датасет собрать нельзя; для записей измерений цели пока нет.</p>');
  const timeRows = CTOR_TIME_PARAMS.map((d) => ctorParamRow("time", 0, s.time, d)).join("");
  $("#ctor-pipeline").innerHTML =
    single("system", "Система", ctorCandidates("system")) +
    list("channels", "Каналы диссипации", channelButtons) +
    single("initial", "Начальное состояние", ctorCandidates("initial")) +
    single("evolve", "Эволюция", ctorCandidates("evolve")) +
    list("observe", "Наблюдение", observeButtons) +
    ctorCard("Цель", targetBody) +
    ctorCard("Время", timeRows);
  ctorRefresh();
}

// ---- keeping the state consistent ----

// After the evolve block changes, observe producers and the target must take
// the output of the new block.
function ctorReconcile() {
  const output = ctorOutput();
  const first = ctorCandidates("observe", "producer")[0];
  ctorState.observe = ctorState.observe.map((it) => {
    const def = ctorBlock(it.fn);
    return def.role === "producer" && def.needs !== output ? ctorNewItem(first.fn) : it;
  });
  if (ctorState.target && ctorBlock(ctorState.target.fn).needs !== output) {
    ctorState.target = null;
  }
}

function ctorChangeBlock(stage, idx, fn) {
  if (stage === "target" && fn === "") {
    ctorState.target = null;
  } else if (stage === "channels" || stage === "observe") {
    ctorState[stage][idx] = ctorNewItem(fn);
  } else {
    ctorState[stage] = ctorNewItem(fn);
  }
  if (stage === "evolve") ctorReconcile();
}

function ctorAction(stage, idx, action) {
  const list = ctorState[stage];
  const role = action === "add-modifier" ? "modifier" : "producer";
  if (action === "add-producer" || action === "add-modifier") {
    const options = ctorCandidates(stage, role);
    if (!options.length) throw new Error(`для стадии ${stage} нет блоков роли ${role}`);
    list.push(ctorNewItem(options[0].fn));
  } else if (action === "remove") {
    list.splice(idx, 1);
  } else if (action === "up" || action === "down") {
    const other = action === "up" ? idx - 1 : idx + 1;
    [list[idx], list[other]] = [list[other], list[idx]];
  }
}

// ---- recipe JSON and names ----

function ctorRefresh() {
  const names = ctorParamNames();
  $$("[data-pname]").forEach((el) => {
    el.textContent = names[el.dataset.pname] ? `→ Param("${names[el.dataset.pname]}")` : "";
  });
  $("#ctor-json").textContent = JSON.stringify(ctorBuildRecipe(), null, 2);
}

function ctorShowError(message) {
  const box = $("#ctor-error");
  box.textContent = message;
  box.classList.remove("hidden");
}

function ctorClearError() {
  $("#ctor-error").classList.add("hidden");
  $("#ctor-error").textContent = "";
}

function ctorStatus(text) {
  $("#ctor-status").textContent = text;
}

async function ctorGuard(action) {
  ctorClearError();
  try {
    await action();
  } catch (err) {
    ctorShowError(err.message || String(err));
  }
}

// ---- preview chart ----

function ctorFmt(x) {
  return Math.abs(x) >= 100 || (x !== 0 && Math.abs(x) < 0.01) ? x.toExponential(1) : x.toFixed(2);
}

function ctorDrawPreview() {
  const svg = $("#ctor-svg");
  if (!ctorPreview) {
    svg.innerHTML = "";
    return;
  }
  const name = $("#ctor-series").value;
  const rows = ctorPreview.trajectories;
  const w = 640;
  const h = 260;
  const pad = 36;
  const xs = rows.flatMap((r) => r.times);
  const ys = rows.flatMap((r) => r.series[name]).filter((y) => y !== null);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
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
  let body = `
    <line x1="${pad}" y1="10" x2="${pad}" y2="${h - pad}" stroke="var(--color-neutral-400)" stroke-width="1"></line>
    <line x1="${pad}" y1="${h - pad}" x2="${w - 10}" y2="${h - pad}" stroke="var(--color-neutral-400)" stroke-width="1"></line>
    <text x="4" y="${sy(ymax) + 4}" font-size="10" fill="var(--color-neutral-700)" font-family="monospace">${ctorFmt(ymax)}</text>
    <text x="4" y="${sy(ymin) + 4}" font-size="10" fill="var(--color-neutral-700)" font-family="monospace">${ctorFmt(ymin)}</text>
    <text x="${pad}" y="${h - 18}" font-size="10" fill="var(--color-neutral-700)" font-family="monospace">${ctorFmt(xmin)}</text>
    <text x="${w - 40}" y="${h - 18}" font-size="10" fill="var(--color-neutral-700)" font-family="monospace">${ctorFmt(xmax)}</text>`;
  rows.forEach((r, i) => {
    const color = CTOR_COLORS[i % CTOR_COLORS.length];
    const pts = r.times
      .map((t, k) => (r.series[name][k] === null ? null : `${sx(t).toFixed(1)},${sy(r.series[name][k]).toFixed(1)}`))
      .filter((p) => p !== null).join(" ");
    body += `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6"></polyline>`;
    if (r.target) {
      const x = sx(r.target.time);
      body += `<line x1="${x}" y1="10" x2="${x}" y2="${h - pad}" stroke="${color}" stroke-width="1.5" stroke-dasharray="3 3"></line>`;
      body += `<text x="${x + 4}" y="${22 + 12 * i}" font-size="11" fill="${color}" font-family="monospace">T=${r.target.time.toFixed(2)}${r.target.censored ? "+" : ""}</text>`;
    }
    body += `<text x="${pad + 8 + 34 * i}" y="${h - 4}" font-size="11" fill="${color}" font-family="system-ui">● ${r.index}</text>`;
  });
  svg.innerHTML = body;
}

function ctorRenderPreview() {
  const p = ctorPreview;
  const select = $("#ctor-series");
  const previous = select.value;
  select.innerHTML = p.series_names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  if (p.series_names.includes(previous)) select.value = previous;
  const fmt = (v) => (typeof v === "number" ? v.toFixed(3) : JSON.stringify(v));
  $("#ctor-table tbody").innerHTML = p.trajectories.map((r, i) => {
    const params = Object.keys(r.params).length
      ? Object.entries(r.params).map(([k, v]) => `${esc(k)}=${esc(fmt(v))}`).join(", ") : "—";
    const t = r.target ? r.target.time.toFixed(3) : "—";
    const c = r.target ? (r.target.censored ? "да" : "нет") : "—";
    return `<tr><td><span style="color:${CTOR_COLORS[i % CTOR_COLORS.length]}">●</span> ${r.index}</td>
      <td class="mono">${params}</td><td>${t}</td><td>${c}</td></tr>`;
  }).join("");
  $("#ctor-elapsed").textContent =
    `Время расчёта: ${p.elapsed_s.toFixed(2)} с · n=${p.n} · seed=${p.seed}`;
  ctorDrawPreview();
}

// ---- actions ----

async function ctorValidate() {
  const data = await api("/api/recipe/validate", {
    method: "POST", body: JSON.stringify({ recipe: ctorBuildRecipe() }),
  });
  if (data.ok) ctorStatus("Рецепт корректен.");
  else ctorShowError(data.error);
}

async function ctorRunPreview() {
  ctorStatus("Считаю…");
  try {
    ctorPreview = await api("/api/recipe/preview", {
      method: "POST",
      body: JSON.stringify({
        recipe: ctorBuildRecipe(),
        n: Number($("#ctor-n").value),
        seed: Number($("#ctor-seed").value),
      }),
    });
  } catch (err) {
    ctorStatus("");
    throw err;
  }
  ctorStatus("");
  ctorRenderPreview();
}

async function ctorLoadList() {
  const data = await api("/api/recipes");
  const group = (label, names) => (names.length
    ? `<optgroup label="${label}">${names.map((n) =>
      `<option value="${esc(n)}">${esc(n)}</option>`).join("")}</optgroup>` : "");
  $("#ctor-load-name").innerHTML = group("примеры", data.examples) + group("сохранённые", data.saved);
}

async function ctorLoad(name) {
  const data = await api(`/api/recipes/${encodeURIComponent(name)}`);
  ctorState = ctorFromRecipe(data.recipe);
  ctorPreview = null;
  $("#ctor-name").value = data.name;
  ctorRenderPipeline();
  ctorDrawPreview();
  ctorStatus(`Загружен рецепт «${data.name}».`);
}

async function ctorSave() {
  const name = $("#ctor-name").value.trim();
  await api("/api/recipes", {
    method: "POST", body: JSON.stringify({ name, recipe: ctorBuildRecipe() }),
  });
  await ctorLoadList();
  $("#ctor-load-name").value = name;
  ctorStatus(`Сохранено: data/recipes/${name}.json`);
}

function ctorStopPoll() {
  if (ctorPoll !== null) clearInterval(ctorPoll);
  ctorPoll = null;
}

async function ctorRefreshJob() {
  const job = await api("/api/jobs");
  renderJob($("#ctor-job-log"), job);
  if (job.status !== "running") ctorStopPoll();
  return job;
}

async function ctorGenerate() {
  const name = $("#ctor-name").value.trim();
  const num = (id) => Number($(id).value);
  const job = await api("/api/recipe/dataset", {
    method: "POST",
    body: JSON.stringify({
      name,
      recipe: ctorBuildRecipe(),
      n: num("#ctor-ds-n"),
      seed: num("#ctor-ds-seed"),
      window_length: num("#ctor-ds-window"),
      samples_per_trajectory: num("#ctor-ds-samples"),
      horizon: num("#ctor-ds-horizon"),
    }),
  });
  renderJob($("#ctor-job-log"), job);
  ctorStatus(`${job.message} → ${job.output}`);
  await ctorLoadList();
  ctorStopPoll();
  ctorPoll = setInterval(() => ctorGuard(ctorRefreshJob), 2000);
}

async function ctorCopy() {
  await navigator.clipboard.writeText($("#ctor-json").textContent);
  ctorStatus("Рецепт скопирован.");
}

// ---- events ----

function ctorOnChange(event) {
  const el = event.target;
  const { stage, idx, field, param } = el.dataset;
  if (!stage || !field) return;
  if ((field === "fn" || field === "vary") && event.type !== "change") return;
  ctorGuard(() => {
    const item = ctorItem(stage, Number(idx));
    if (field === "fn") {
      ctorChangeBlock(stage, Number(idx), el.value);
      ctorRenderPipeline();
      return;
    }
    const st = item.p[param];
    if (field === "checked") st.checked = el.checked;
    else if (field === "vary") {
      st.vary = el.checked;
      if (st.vary && st.lo === "" && st.hi === "") {
        const v = Number(st.raw);
        const known = st.raw.trim() !== "" && Number.isFinite(v);
        st.lo = String(known ? (v > 0 ? v * 0.5 : v - 1) : 0);
        st.hi = String(known ? (v > 0 ? v * 1.5 : v + 1) : 1);
      }
      ctorRenderPipeline();
      return;
    } else st[field] = el.value;
    ctorRefresh();
  });
}

function ctorOnClick(event) {
  const el = event.target.closest("button[data-action]");
  if (!el || el.disabled) return;
  ctorGuard(() => {
    ctorAction(el.dataset.stage, Number(el.dataset.idx), el.dataset.action);
    ctorRenderPipeline();
  });
}

async function ctorBoot() {
  ctorCatalog = await api("/api/blocks");
  ctorState = ctorSkeleton();
  const pipeline = $("#ctor-pipeline");
  pipeline.addEventListener("input", ctorOnChange);
  pipeline.addEventListener("change", ctorOnChange);
  pipeline.addEventListener("click", ctorOnClick);
  $("#ctor-series").addEventListener("change", ctorDrawPreview);
  $("#btn-ctor-validate").addEventListener("click", () => ctorGuard(ctorValidate));
  $("#btn-ctor-preview").addEventListener("click", () => ctorGuard(ctorRunPreview));
  $("#btn-ctor-copy").addEventListener("click", () => ctorGuard(ctorCopy));
  $("#btn-ctor-save").addEventListener("click", () => ctorGuard(ctorSave));
  $("#btn-ctor-load").addEventListener("click", () =>
    ctorGuard(() => ctorLoad($("#ctor-load-name").value)));
  $("#btn-ctor-gen").addEventListener("click", () => ctorGuard(ctorGenerate));
  $("#btn-ctor-job").addEventListener("click", () => ctorGuard(ctorRefreshJob));
  ctorRenderPipeline();
  await ctorLoadList();
  await ctorLoad("scenario_b_shots");
}

ctorBoot().catch((err) => {
  console.error(err);
  ctorShowError(`Конструктор не загрузился: ${err.message}`);
});
