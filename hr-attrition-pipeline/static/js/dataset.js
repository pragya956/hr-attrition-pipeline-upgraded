(async function () {
  const { $, $$, api, esc, fmt, counter, loaded, empty, toast, chart } = App;
  const state = { page: 1, sort_by: "", sort_dir: "asc", columns: null, allColumns: [] };
  const filterKeys = ["department", "job_role", "gender", "overtime", "attrition"];
  let debounce;

  $$(".tab").forEach((t) => t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.setAttribute("aria-selected", x === t));
    $$("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== t.dataset.tab));
    if (t.dataset.tab === "dups") loadDups();
    if (t.dataset.tab === "charts" && !App.charts["ch-col"]) drawColumn();
  }));

  function query() {
    const p = new URLSearchParams({ page: state.page, page_size: $("#f-size").value });
    if ($("#f-search").value.trim()) p.set("search", $("#f-search").value.trim());
    filterKeys.forEach((k) => { const v = $("#f-" + k).value; if (v) p.set(k, v); });
    if (state.sort_by) { p.set("sort_by", state.sort_by); p.set("sort_dir", state.sort_dir); }
    if (state.columns) p.set("columns", state.columns.join(","));
    return p;
  }
  function syncDownload() { const p = query(); ["page", "page_size", "sort_by", "sort_dir", "columns"].forEach((k) => p.delete(k)); $("#download-btn").href = "/api/dataset/download?" + p; }

  async function loadTable() {
    syncDownload();
    try {
      const d = await api("/api/dataset/preview?" + query());
      state.page = d.page;
      $("#thead").innerHTML = "<tr>" + d.columns.map((c) => `<th class="sortable ${state.sort_by === c ? "sorted" : ""}" data-col="${esc(c)}" tabindex="0" aria-sort="${state.sort_by === c ? (state.sort_dir === "asc" ? "ascending" : "descending") : "none"}">${esc(c)}<span class="arrow">${state.sort_by === c && state.sort_dir === "desc" ? "▼" : "▲"}</span></th>`).join("") + "</tr>";
      $("#tbody").innerHTML = d.rows.length
        ? d.rows.map((r) => "<tr>" + d.columns.map((c) => { const v = r[c]; if (c === "Attrition") return `<td><span class="badge ${v === "Yes" ? "red" : "green"}">${v === "Yes" ? "Left" : "Stayed"}</span></td>`; return `<td class="${typeof v === "number" ? "num" : ""}">${v === null ? "—" : esc(v)}</td>`; }).join("") + "</tr>").join("")
        : `<tr><td colspan="${d.columns.length}"><div class="empty"><i class="bi bi-search"></i><h3>No matching employees</h3><p>Try clearing the search or filters.</p></div></td></tr>`;
      $("#pager-info").textContent = `${fmt(d.total)} matching employees`;
      $("#pg-num").textContent = `Page ${d.page} of ${d.pages}`;
      $("#pg-prev").disabled = d.page <= 1; $("#pg-next").disabled = d.page >= d.pages;
    } catch (e) { toast(e.message, "error"); }
  }
  $("#thead").addEventListener("click", (e) => { const th = e.target.closest("th"); if (!th) return; sortBy(th.dataset.col); });
  $("#thead").addEventListener("keydown", (e) => { if (e.key === "Enter") { const th = e.target.closest("th"); if (th) sortBy(th.dataset.col); } });
  function sortBy(c) { if (state.sort_by === c) state.sort_dir = state.sort_dir === "asc" ? "desc" : "asc"; else { state.sort_by = c; state.sort_dir = "asc"; } state.page = 1; loadTable(); }
  $("#pg-prev").onclick = () => { state.page--; loadTable(); }; $("#pg-next").onclick = () => { state.page++; loadTable(); };
  $("#f-search").addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(() => { state.page = 1; loadTable(); }, 250); });
  [...filterKeys.map((k) => "#f-" + k), "#f-size"].forEach((s) => $(s).addEventListener("change", () => { state.page = 1; loadTable(); }));
  $("#btn-reset").onclick = () => { $("#f-search").value = ""; filterKeys.forEach((k) => ($("#f-" + k).value = "")); state.sort_by = ""; state.page = 1; loadTable(); };

  // ----- summary + columns
  try {
    const [s, c] = await Promise.all([api("/api/dataset/summary"), api("/api/dataset/columns")]);
    $$("#ds-kpis .kpi").forEach(loaded);
    $("#ds-file").textContent = s.filename; $("#ds-source").textContent = "source: " + s.source;
    counter($("#ds-rows"), s.rows); counter($("#ds-cols"), s.columns); counter($("#ds-missing"), s.missing_total); counter($("#ds-dups"), s.duplicates.full_rows);
    $("#ds-dups-hint").textContent = `${s.duplicates.ignoring_identifier} ignoring identifier`;
    counter($("#ds-target"), s.target.rate, 1, "%"); $("#ds-target-hint").textContent = `${fmt(s.target.counts[1])} Yes · ${fmt(s.target.counts[0])} No`;
    filterKeys.forEach((k) => ($("#f-" + k).innerHTML += (s.filter_options[k] || []).map((o) => `<option>${esc(o)}</option>`).join("")));
    state.allColumns = s.column_names;
    $("#col-picker").innerHTML = s.column_names.map((n) => `<label class="switch"><input type="checkbox" checked value="${esc(n)}"> ${esc(n)}</label>`).join("");
    $("#col-picker").addEventListener("change", () => { const on = $$("#col-picker input:checked").map((i) => i.value); state.columns = on.length === state.allColumns.length ? null : on; if (!on.length) { toast("Select at least one column.", "error"); return; } loadTable(); });
    $("#cols-body").innerHTML = c.columns.map((x) => `<tr><td><strong>${esc(x.name)}</strong></td><td class="mono">${esc(x.dtype)}</td><td><span class="badge ${x.kind === "numeric" ? "blue" : "amber"}">${x.kind}</span></td><td class="num">${x.missing ? `<span class="badge red">${x.missing}</span>` : 0}</td><td class="num">${x.unique}</td></tr>`).join("");
    $("#num-body").innerHTML = Object.entries(c.numeric).map(([k, v]) => `<tr><td><strong>${esc(k)}</strong></td>${["count", "mean", "std", "min", "25%", "50%", "75%", "max"].map((m) => `<td class="num">${fmt(v[m], m === "count" ? 0 : 2)}</td>`).join("")}</tr>`).join("");
    $("#cat-body").innerHTML = Object.entries(c.categorical).map(([k, v]) => `<tr><td><strong>${esc(k)}</strong></td><td class="num">${v.unique}</td><td style="white-space:normal">${v.top.slice(0, 6).map((t, i) => `<span class="badge gray">${esc(t)} (${v.counts[i]})</span>`).join(" ")}</td></tr>`).join("");
    $("#cc-col").innerHTML = c.columns.map((x) => `<option value="${esc(x.name)}">${esc(x.name)}</option>`).join("");
    $("#cc-col").value = "Department"; $("#cc-col").onchange = drawColumn;
  } catch (e) {
    $$("#ds-kpis .kpi").forEach(loaded);
    empty($("#table-wrap"), "Dataset unavailable", e.message, "bi-cloud-slash", true); return;
  }
  loadTable();

  async function drawColumn() {
    try { const d = await api("/api/dataset/column-chart?column=" + encodeURIComponent($("#cc-col").value)); Viz.stacked("ch-col", d); } catch (e) { toast(e.message, "error"); }
  }
  async function loadDups() {
    const box = $("#dups-box");
    try {
      const d = await api("/api/dataset/duplicates");
      if (!d.count) { box.innerHTML = `<div class="empty"><i class="bi bi-check2-circle"></i><h3>No duplicate records</h3><p>Every row is unique, even when the identifier is ignored.</p></div>`; return; }
      box.innerHTML = `<p>${d.count} rows belong to duplicate groups (first ${d.rows.length} shown).</p><div class="table-wrap"><table><thead><tr>${d.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${d.rows.map((r) => `<tr>${d.columns.map((c) => `<td>${esc(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    } catch (e) { empty(box, "Could not load duplicates", e.message, "bi-exclamation-triangle", true); }
  }
})();
