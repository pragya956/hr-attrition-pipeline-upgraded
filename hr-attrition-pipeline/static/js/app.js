/* Shared helpers: API client, toasts, counters, chart defaults, status polling, shell behaviour. */
const App = (() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n, d = 0) => (n === null || n === undefined || Number.isNaN(n)) ? "—" : Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (n, d = 1) => (n === null || n === undefined) ? "—" : (n * 100).toFixed(d) + "%";
  const PALETTE = ["#3b5bfd", "#7c4dff", "#12b8d6", "#ff6b6b", "#f5a524", "#17b26a", "#e056a0", "#6c7a9c"];

  class ApiError extends Error { constructor(msg, status, body) { super(msg); this.status = status; this.body = body; } }
  async function api(path, opts = {}) {
    const init = { headers: {}, ...opts };
    if (opts.json !== undefined) { init.method = init.method || "POST"; init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.json); }
    let res;
    try { res = await fetch(path, init); } catch (e) { throw new ApiError("Cannot reach the server. Is the Flask app running?", 0); }
    let body = null;
    try { body = await res.json(); } catch (e) { /* non-JSON */ }
    if (!res.ok) throw new ApiError((body && body.error) || `Request failed (${res.status})`, res.status, body);
    return body;
  }

  function toast(message, type = "info", ms = 4500) {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    const icon = { success: "bi-check-circle-fill", error: "bi-exclamation-octagon-fill", info: "bi-info-circle-fill" }[type] || "bi-info-circle-fill";
    el.innerHTML = `<i class="bi ${icon}"></i><div>${esc(message)}</div>`;
    $("#toasts").appendChild(el);
    setTimeout(() => { el.style.transition = "opacity .3s"; el.style.opacity = 0; setTimeout(() => el.remove(), 300); }, ms);
  }

  function counter(el, value, decimals = 0, suffix = "") {
    if (value === null || value === undefined) { el.textContent = "—"; return; }
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { el.textContent = fmt(value, decimals) + suffix; return; }
    const t0 = performance.now(), dur = 900;
    const step = (t) => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3); el.textContent = fmt(value * e, decimals) + suffix; if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }

  function applyChartTheme() {
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.color = css("--text-2");
    Chart.defaults.borderColor = css("--border");
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
    Chart.defaults.plugins.tooltip.backgroundColor = "#0f1635";
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.maintainAspectRatio = false;
  }
  const charts = {};
  function chart(id, config) {
    const canvas = document.getElementById(id);
    if (!canvas) return null;
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(canvas, config);
    return charts[id];
  }
  const loaded = (el) => el && el.classList.remove("skeleton");
  function empty(el, title, text, icon = "bi-inbox", error = false) {
    el.innerHTML = `<div class="empty ${error ? "error" : ""}"><i class="bi ${icon}"></i><h3>${esc(title)}</h3><p>${esc(text)}</p></div>`;
  }
  function setBusy(btn, busy) { btn.classList.toggle("loading", busy); btn.disabled = busy; }

  /* ---- status pills ---- */
  let lastStatus = null;
  const listeners = [];
  function paintStatus(s) {
    lastStatus = s;
    const d = s.dataset, m = s.model, t = s.training;
    const set = (id, cls, text) => { const p = $(id); p.querySelector(".dot")?.setAttribute("class", "dot " + cls); p.querySelector(".txt").textContent = text; };
    if (d.state === "ready") set("#pill-dataset", "ok", `Dataset: ${fmt(d.rows)} rows`);
    else if (d.state === "error") set("#pill-dataset", "bad", "Dataset: unavailable");
    else set("#pill-dataset", "warn", "Dataset: loading…");
    if (t.status === "running") set("#pill-model", "warn", `Training ${Math.round(t.progress * 100)}%`);
    else if (t.status === "failed") set("#pill-model", "bad", "Training failed");
    else if (m.available) set("#pill-model", "ok", `Model: ${m.name}`);
    else set("#pill-model", "bad", "Model: not trained");
    const tr = $("#pill-trained .txt");
    tr.textContent = m.trained_at ? new Date(m.trained_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "Not trained";
    listeners.forEach((fn) => fn(s));
  }
  async function pollStatus() {
    let delay = 5000;
    try { const s = await api("/api/model/status"); paintStatus(s); if (s.training.status === "running" || s.dataset.state === "loading") delay = 1500; }
    catch (e) { /* keep last pill */ }
    setTimeout(pollStatus, delay);
  }
  function onStatus(fn) { listeners.push(fn); if (lastStatus) fn(lastStatus); }

  function initShell() {
    const app = $("#app");
    $("#menu-btn").addEventListener("click", () => {
      if (matchMedia("(max-width: 900px)").matches) app.classList.toggle("nav-open"); else app.classList.toggle("collapsed");
    });
    $("#scrim").addEventListener("click", () => app.classList.remove("nav-open"));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") app.classList.remove("nav-open"); });
    $("#theme-btn").addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* private mode */ }
      applyChartTheme(); Object.values(charts).forEach((c) => c.update());
      $("#theme-btn i").className = next === "dark" ? "bi bi-sun" : "bi bi-moon-stars";
    });
    if (document.documentElement.getAttribute("data-theme") === "dark") $("#theme-btn i").className = "bi bi-sun";
    applyChartTheme();
    pollStatus();
  }
  document.addEventListener("DOMContentLoaded", initShell);
  return { $, $$, api, toast, counter, chart, charts, esc, fmt, pct, empty, loaded, setBusy, onStatus, PALETTE, css, ApiError };
})();

/* Reusable chart builders ------------------------------------------------ */
const Viz = {
  /** Stacked counts (stayed vs left) with attrition-rate in the tooltip. */
  stacked(id, d, { horizontal = false } = {}) {
    return App.chart(id, { type: "bar", data: { labels: d.labels, datasets: [
      { label: "Stayed", data: d.stayed, backgroundColor: "#3b5bfd", borderRadius: 4 },
      { label: "Left", data: d.left, backgroundColor: "#ff6b6b", borderRadius: 4 }] },
      options: { indexAxis: horizontal ? "y" : "x", scales: { x: { stacked: true, grid: { display: !horizontal ? false : true } }, y: { stacked: true, beginAtZero: true } },
        plugins: { tooltip: { callbacks: { footer: (items) => `Attrition rate: ${d.rate[items[0].dataIndex]}% of ${d.total[items[0].dataIndex]}` } } } } });
  },
  /** Attrition-rate bars (percentage) with counts in the tooltip. */
  rate(id, d, { horizontal = false, color = "#7c4dff" } = {}) {
    return App.chart(id, { type: "bar", data: { labels: d.labels, datasets: [{ label: "Attrition rate (%)", data: d.rate, backgroundColor: color, borderRadius: 6 }] },
      options: { indexAxis: horizontal ? "y" : "x", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.parsed[horizontal ? "x" : "y"]}% attrition`, footer: (items) => { const i = items[0].dataIndex; return `${d.left[i]} left of ${d.total[i]} employees`; } } } },
        scales: { [horizontal ? "x" : "y"]: { beginAtZero: true, ticks: { callback: (v) => v + "%" } }, [horizontal ? "y" : "x"]: { grid: { display: false } } } } });
  },
};
