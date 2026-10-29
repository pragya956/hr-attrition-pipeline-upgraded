(async function () {
  const { $, api, counter, chart, empty, loaded, fmt, pct } = App;
  try {
    const o = await api("/api/overview");
    const $$kpi = $("#kpis").querySelectorAll(".kpi"); $$kpi.forEach(loaded);
    counter($("#kpi-total"), o.total); counter($("#kpi-left"), o.left); counter($("#kpi-stayed"), o.stayed);
    counter($("#kpi-rate"), o.rate, 1, "%"); counter($("#kpi-features"), o.n_features);
    if (o.cv && o.cv.mean !== null) {
      const label = o.cv.metric.replace(/_/g, " ");
      $("#kpi-cv-label").textContent = `CV ${label}`;
      counter($("#kpi-cv"), o.cv.mean, 3);
      $("#kpi-cv-hint").textContent = `${o.cv.model} · ±${fmt(o.cv.std, 3)}`;
    }
    chart("ch-dist", { type: "doughnut", data: { labels: o.distribution.labels, datasets: [{ data: o.distribution.values, backgroundColor: ["#3b5bfd", "#ff6b6b"], borderWidth: 0, hoverOffset: 10 }] },
      options: { cutout: "68%", plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => `${c.label}: ${fmt(c.parsed)} (${(100 * c.parsed / o.total).toFixed(1)}%)` } } } } });
    Viz.stacked("ch-dept", o.department);
    Viz.rate("ch-sat", o.job_satisfaction, { color: "#12b8d6" });
    Viz.rate("ch-ot", o.overtime, { color: "#ff6b6b" });
    chart("ch-income", { type: "bar", data: { labels: o.income.labels.map((l) => (l === "Yes" ? "Left" : "Stayed")), datasets: [
      { label: "Mean", data: o.income.mean, backgroundColor: "#3b5bfd", borderRadius: 6 }, { label: "Median", data: o.income.median, backgroundColor: "#7c4dff", borderRadius: 6 }] },
      options: { scales: { y: { beginAtZero: true, ticks: { callback: (v) => "$" + fmt(v) } } }, plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: $${fmt(c.parsed.y)}` } } } } });
    Viz.rate("ch-age", o.age, { color: "#7c4dff" });
  } catch (e) {
    App.$$(".kpi").forEach(loaded);
    App.$$(".chart-box").forEach((b) => empty(b, "Dataset unavailable", e.message, "bi-cloud-slash", true));
    return;
  }
  try {
    const fi = await api("/api/models/feature-importance");
    const items = fi.permutation.items.slice(0, 10).reverse();
    chart("ch-fi", { type: "bar", data: { labels: items.map((i) => i.feature), datasets: [{ label: "Drop in ROC-AUC when shuffled", data: items.map((i) => i.mean), backgroundColor: "#7c4dff", borderRadius: 5 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { y: { grid: { display: false } } } } });
    $("#fi-sub").textContent = `${fi.permutation.model} · permutation importance on held-out data (association, not causation)`;
  } catch (e) {
    empty($("#fi-box"), "No model yet", e.status === 409 ? "Feature importance appears once a model has been trained." : e.message, "bi-cpu");
  }
})();
