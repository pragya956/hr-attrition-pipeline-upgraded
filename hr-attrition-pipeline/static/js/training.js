(function () {
  const { $, api, esc, fmt, toast, setBusy, onStatus, pct } = App;
  let running = false;
  $("#t-folds").addEventListener("input", () => ($("#t-folds-val").textContent = $("#t-folds").value));

  function renderResults() {
    api("/api/models/metrics").then((m) => {
      const names = Object.keys(m.cv_results), sm = m.selection_metric;
      $("#t-cv-sub").textContent = `${m.params.n_splits}-fold stratified CV · mean ± std`;
      const cols = ["accuracy", "balanced_accuracy", "recall", "f1", "roc_auc", "average_precision"];
      $("#t-cv").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Model</th>${cols.map((c) => `<th>${c.replace(/_/g, " ")}</th>`).join("")}</tr></thead><tbody>${names.map((n) => `<tr><td><strong>${esc(n)}</strong> ${n === m.model_name ? '<span class="badge green">selected</span>' : ""}${m.cv_results[n].failed_folds ? `<span class="badge red">${m.cv_results[n].failed_folds} failed</span>` : ""}</td>${cols.map((c) => { const r = m.cv_results[n][c]; return `<td class="num" ${c === sm ? 'style="font-weight:700"' : ""}>${r.mean === null ? "—" : fmt(r.mean, 3) + ' <small style="color:var(--text-3)">±' + fmt(r.std, 3) + "</small>"}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div>`;
      const t = m.test_metrics;
      $("#t-sel").innerHTML = `<p><strong>${esc(m.model_name)}</strong> had the best mean CV <strong>${sm.replace(/_/g, " ")}</strong> (${fmt(m.cv_results[m.model_name][sm].mean, 3)}). It was then scored once on ${m.split.test_rows} held-out employees (seed ${m.seed}).</p>
        <div class="grid" style="grid-template-columns:repeat(3,1fr);gap:10px">${[["Recall", t.recall], ["Precision", t.precision], ["F1", t.f1], ["ROC-AUC", t.roc_auc], ["PR-AUC", t.average_precision], ["Accuracy", t.accuracy]].map(([k, v]) => `<div class="threshold-box"><div class="label-text">${k}</div><div style="font-size:1.35rem;font-weight:700">${v === null ? "—" : fmt(v, 3)}</div></div>`).join("")}</div>
        <p class="mt" style="color:var(--text-2);font-size:.85rem">Always predicting “stayed” would score ${pct(t.majority_baseline_accuracy)} accuracy, so accuracy alone says little. <a href="/evaluation">See full evaluation</a>.</p>`;
    }).catch(() => {});
  }

  onStatus((s) => {
    const t = s.training, bar = $("#t-bar");
    bar.firstElementChild.style.width = Math.round(t.progress * 100) + "%";
    bar.classList.toggle("running", t.status === "running");
    const st = $("#t-state");
    const map = { idle: ["", "Idle"], running: ["warn", `Running · ${Math.round(t.progress * 100)}%`], completed: ["ok", "Completed"], failed: ["bad", "Failed"] }[t.status];
    st.querySelector(".dot").className = "dot " + map[0]; st.querySelector(".txt").textContent = map[1];
    api("/api/model/status").then((full) => {
      const log = $("#t-log"), near = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
      log.innerHTML = full.training.logs.map((l) => `<div class="${/failed|error|WARNING/i.test(l) ? "err" : ""}">${esc(l)}</div>`).join("");
      if (near) log.scrollTop = log.scrollHeight;
    }).catch(() => {});
    setBusy($("#t-start"), t.status === "running");
    if (running && t.status !== "running") {
      running = false;
      if (t.status === "completed") { toast("Training finished. Model saved.", "success"); } else toast(t.error || "Training failed.", "error");
    }
    if (t.status === "running") running = true;
    if (s.model.available && t.status !== "running") renderResults();
  });

  $("#t-start").addEventListener("click", async () => {
    const seed = Number($("#t-seed").value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) { $("#t-seed").classList.add("invalid"); $("#t-seed-err").textContent = "Enter a whole number between 0 and 2,147,483,647."; return; }
    $("#t-seed").classList.remove("invalid"); $("#t-seed-err").textContent = "";
    setBusy($("#t-start"), true);
    try {
      await api("/api/models/train", { json: { n_splits: Number($("#t-folds").value), seed, class_weight: $("#t-cw").checked, tune: $("#t-tune").checked, selection_metric: $("#t-metric").value } });
      running = true; toast("Training started.", "info");
    } catch (e) { toast(e.message, e.status === 409 ? "info" : "error"); setBusy($("#t-start"), false); }
  });
})();
