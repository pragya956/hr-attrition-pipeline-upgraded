(async function () {
  const { $, $$, api, esc, fmt, chart, empty } = App;
  let d;
  try { d = await api("/api/models/feature-importance"); }
  catch (e) { $("#fi-body").hidden = true; const b = $("#fi-empty"); b.hidden = false; empty(b, e.status === 409 ? "No trained model" : "Could not load importances", e.status === 409 ? "Train a model to see which features it relies on." : e.message, "bi-sliders", e.status !== 409); return; }
  $$(".tab").forEach((t) => t.addEventListener("click", () => { $$(".tab").forEach((x) => x.setAttribute("aria-selected", x === t)); $$("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== t.dataset.tab)); }));
  const bars = (id, rows, label, val, color, extra) => { const r = rows.slice(0, 15).reverse(); chart(id, { type: "bar", data: { labels: r.map((x) => x[label]), datasets: [{ data: r.map((x) => x[val]), backgroundColor: typeof color === "function" ? r.map(color) : color, borderRadius: 5 }] }, options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: extra ? { label: (c) => extra(r[c.dataIndex]) } : {} } }, scales: { y: { grid: { display: false } } } } }); };
  $("#fi-perm-sub").textContent = `${d.permutation.model} · scoring: ${d.permutation.scoring}`;
  bars("fi-perm", d.permutation.items, "feature", "mean", "#7c4dff", (r) => `mean ${fmt(r.mean, 4)} ± ${fmt(r.std, 4)}`);
  if (d.random_forest) bars("fi-rf", d.random_forest.by_feature, "feature", "importance", "#12b8d6", (r) => `${(r.importance * 100).toFixed(1)}% of total importance`);
  else empty($("#fi-rf").parentElement, "Unavailable", "The random forest could not be fitted.", "bi-tree");
  if (d.logistic_regression) {
    const rows = d.logistic_regression.coefficients;
    bars("fi-lr", rows, "feature", "coefficient", (r) => (r.coefficient > 0 ? "#ff6b6b" : "#3b5bfd"), (r) => `coef ${fmt(r.coefficient, 3)} (odds ×${fmt(r.odds_ratio, 2)})`);
    $("#fi-lr-body").innerHTML = rows.map((r) => `<tr><td>${esc(r.feature)}</td><td class="num">${fmt(r.coefficient, 3)}</td><td class="num">${fmt(r.odds_ratio, 2)}</td><td><span class="badge ${r.coefficient > 0 ? "red" : "blue"}">${r.coefficient > 0 ? "higher attrition odds" : "lower attrition odds"}</span></td></tr>`).join("");
  } else empty($("#fi-lr").parentElement, "Unavailable", "The logistic regression could not be fitted.", "bi-graph-up");
})();
