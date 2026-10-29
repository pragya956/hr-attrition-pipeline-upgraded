(async function () {
  const { $, api, esc, empty, toast } = App;
  const keys = ["department", "overtime", "gender"];
  try {
    const s = await api("/api/dataset/summary");
    keys.forEach((k) => ($("#a-" + k).innerHTML += (s.filter_options[k] || []).map((o) => `<option>${esc(o)}</option>`).join("")));
  } catch (e) { empty($("#a-empty"), "Dataset unavailable", e.message, "bi-cloud-slash", true); $("#a-empty").hidden = false; $("#a-grid").hidden = true; return; }

  const map = [["ac-dept", "by_department", "stacked"], ["ac-ot", "by_overtime", "rate"], ["ac-role", "by_job_role", "rateH"], ["ac-travel", "by_business_travel", "rate"],
    ["ac-age", "by_age_group", "rate"], ["ac-years", "by_years_at_company", "rate"], ["ac-income", "by_income_band", "rate"], ["ac-sat", "by_job_satisfaction", "rate"], ["ac-wlb", "by_work_life_balance", "rate"]];

  async function load() {
    const p = new URLSearchParams();
    keys.forEach((k) => { if ($("#a-" + k).value) p.set(k, $("#a-" + k).value); });
    try {
      const d = await api("/api/analytics/attrition?" + p);
      $("#a-empty").hidden = d.empty ? false : true; $("#a-grid").hidden = !!d.empty;
      if (d.empty) { empty($("#a-empty"), "No employees match", d.message, "bi-funnel"); $("#a-summary .txt").textContent = "0 employees"; return; }
      $("#a-summary .txt").textContent = `${d.overall.total} employees · ${d.overall.left} left · ${d.overall.rate}% attrition`;
      const colors = ["#3b5bfd", "#7c4dff", "#12b8d6", "#ff6b6b", "#f5a524", "#17b26a", "#e056a0", "#6c7a9c", "#3b5bfd"];
      map.forEach(([id, key, kind], i) => {
        const data = d[key], box = document.getElementById(id).parentElement;
        if (!data || !data.labels.length) { App.charts[id]?.destroy(); delete App.charts[id]; box.dataset.empty = 1; return; }
        if (kind === "stacked") Viz.stacked(id, data); else Viz.rate(id, data, { horizontal: kind === "rateH", color: colors[i] });
      });
    } catch (e) { toast(e.message, "error"); }
  }
  keys.forEach((k) => $("#a-" + k).addEventListener("change", load));
  $("#a-reset").onclick = () => { keys.forEach((k) => ($("#a-" + k).value = "")); load(); };
  load();
})();
