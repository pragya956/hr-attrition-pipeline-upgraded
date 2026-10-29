(async function () {
  const { $, $$, api, esc, fmt, toast, setBusy, empty } = App;
  let schema, last = null;
  try { schema = await api("/api/model/schema"); }
  catch (e) { $("#p-body").hidden = true; const b = $("#p-empty"); b.hidden = false; empty(b, e.status === 409 ? "No trained model" : "Could not load the model", e.status === 409 ? "Train a model first, then come back to make predictions." : e.message, "bi-person-x", e.status !== 409); return; }

  $("#p-model").textContent = `Model: ${schema.model_name}`;
  $("#p-thr").value = schema.threshold; $("#p-thr-val").textContent = Number(schema.threshold).toFixed(2);
  const likert = (n) => schema.likert[n];
  $("#p-form").innerHTML = schema.fields.map((f) => {
    const id = "pf-" + f.name; let control;
    if (f.kind === "categorical") control = `<select id="${id}" data-name="${f.name}">${f.options.map((o) => `<option ${o === f.default ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
    else if (f.ordinal) {
      const labels = likert(f.name); const opts = []; for (let v = f.min; v <= f.max; v++) opts.push(v);
      control = `<select id="${id}" data-name="${f.name}">${opts.map((v) => `<option value="${v}" ${v === Math.round(f.default) ? "selected" : ""}>${v}${labels && labels[v] ? " – " + labels[v] : ""}</option>`).join("")}</select>`;
    } else control = `<input id="${id}" data-name="${f.name}" type="number" inputmode="decimal" step="${f.integer ? 1 : "any"}" min="0" value="${f.default}">`;
    const hint = f.kind === "numeric" && !f.ordinal ? `typical range ${fmt(f.min)}–${fmt(f.max)}` : "";
    return `<div class="field"><label for="${id}">${esc(f.label)}</label>${control}<span class="help">${hint}</span><span class="field-error" id="err-${f.name}" role="alert"></span></div>`;
  }).join("");

  const defaults = {}; $$("#p-form [data-name]").forEach((el) => (defaults[el.dataset.name] = el.value));
  $("#p-reset").onclick = () => { $$("#p-form [data-name]").forEach((el) => { el.value = defaults[el.dataset.name]; el.classList.remove("invalid"); }); $$(".field-error").forEach((e) => (e.textContent = "")); last = null;
    $("#p-result").innerHTML = `<div class="empty"><i class="bi bi-person-badge"></i><h3>No prediction yet</h3><p>Fill in the profile and press “Predict attrition”.</p></div>`; };

  function gauge(p, thr) {
    const color = p >= 0.66 ? "#ff6b6b" : p >= 0.33 ? "#f5a524" : "#17b26a";
    const ang = Math.PI * (1 - thr), tx = 110 + 100 * Math.cos(ang), ty = 110 - 100 * Math.sin(ang), tx2 = 110 + 82 * Math.cos(ang), ty2 = 110 - 82 * Math.sin(ang);
    return `<div class="gauge"><svg viewBox="0 0 220 124" role="img" aria-label="Attrition probability ${(p * 100).toFixed(1)} percent"><path d="M10 110 A100 100 0 0 1 210 110" pathLength="100" fill="none" stroke="var(--surface-2)" stroke-width="16" stroke-linecap="round"/><path d="M10 110 A100 100 0 0 1 210 110" pathLength="100" fill="none" stroke="${color}" stroke-width="16" stroke-linecap="round" stroke-dasharray="${Math.max(0.5, p * 100)} 100" style="transition:stroke-dasharray .6s"/><line x1="${tx2}" y1="${ty2}" x2="${tx}" y2="${ty}" stroke="var(--text)" stroke-width="2.5"/></svg><div class="gauge-val">${(p * 100).toFixed(1)}%<span class="gauge-sub">probability of attrition</span></div></div>`;
  }
  function render(r) {
    const cls = r.predicted_class ? "red" : "green";
    $("#p-result").innerHTML = `${gauge(r.probability, r.threshold)}
      <div class="result-class">${esc(r.prediction)}</div>
      <span class="badge ${cls}">${r.risk_band} estimated risk</span> <span class="badge gray">threshold ${r.threshold.toFixed(2)}</span>
      <p style="margin:14px 0 0;color:var(--text-2);font-size:.86rem;text-align:left">The ${esc(r.model)} model estimates a ${(r.probability * 100).toFixed(1)}% chance that employees with this profile left in the historical data. This is a probability, not a certainty.</p>
      ${r.drivers.length ? `<div style="text-align:left;margin-top:14px"><div class="label-text">What moves this estimate</div><p class="help" style="font-size:.75rem;color:var(--text-3);margin:2px 0 6px">Change in probability versus the typical value of each field. Association, not cause.</p>${r.drivers.map((d) => `<div class="driver"><div>${esc(d.feature)} = ${esc(typeof d.value === "number" ? fmt(d.value, 0) : d.value)}<small>typical: ${esc(typeof d.typical === "number" ? fmt(d.typical, 0) : d.typical)}</small></div><span class="eff ${d.effect > 0 ? "up" : "down"}">${d.effect > 0 ? "+" : ""}${(d.effect * 100).toFixed(1)} pts</span></div>`).join("")}</div>` : ""}
      ${r.warnings.map((w) => `<div class="callout warn" style="margin-top:10px;text-align:left"><i class="bi bi-exclamation-triangle"></i><div>${esc(w)}</div></div>`).join("")}
      <p style="margin-top:14px;font-size:.75rem;color:var(--text-3);text-align:left">${esc(r.disclaimer)}</p>`;
  }

  $("#p-submit").onclick = async () => {
    $$(".field-error").forEach((e) => (e.textContent = "")); $$("#p-form .invalid").forEach((e) => e.classList.remove("invalid"));
    const body = {}; let bad = false;
    $$("#p-form [data-name]").forEach((el) => {
      const n = el.dataset.name, f = schema.fields.find((x) => x.name === n);
      if (el.value.trim() === "") return;
      if (f.kind === "numeric") { const v = Number(el.value); if (!Number.isFinite(v) || v < 0) { el.classList.add("invalid"); $("#err-" + n).textContent = "Enter a non-negative number."; bad = true; return; } body[n] = v; }
      else body[n] = el.value;
    });
    if (bad) { toast("Please correct the highlighted fields.", "error"); return; }
    body._threshold = Number($("#p-thr").value);
    setBusy($("#p-submit"), true);
    try { last = await api("/api/predict", { json: body }); render(last); }
    catch (e) {
      if (e.status === 422 && e.body && e.body.fields) { Object.entries(e.body.fields).forEach(([k, msg]) => { const er = $("#err-" + k), inp = $("#pf-" + k); if (er) er.textContent = msg; if (inp) inp.classList.add("invalid"); }); toast("Some inputs are invalid.", "error"); }
      else toast(e.message, "error");
    } finally { setBusy($("#p-submit"), false); }
  };
  $("#p-thr").addEventListener("input", () => { $("#p-thr-val").textContent = Number($("#p-thr").value).toFixed(2); if (last) $("#p-submit").click(); });
  $("#p-thr-save").onclick = async () => { try { await api("/api/model/threshold", { json: { threshold: Number($("#p-thr").value) } }); toast("Default threshold saved (no retraining needed).", "success"); } catch (e) { toast(e.message, "error"); } };
})();
