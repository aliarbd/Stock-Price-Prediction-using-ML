const form = document.getElementById("run-form");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const progressStage = document.getElementById("progress-stage");
const statusAlert = document.getElementById("status-alert");
const resultsRoot = document.getElementById("results-root");

const downloadCsvBtn = document.getElementById("download-csv");
const downloadExcelBtn = document.getElementById("download-excel");
const downloadReportBtn = document.getElementById("download-report");

const selectAllBtn = document.getElementById("select-all-models");
const clearAllBtn = document.getElementById("clear-all-models");

let lastResult = null;

function setStatus(message, kind = "info") {
  statusAlert.className = `alert alert-${kind} mt-3`;
  statusAlert.textContent = message;
  statusAlert.classList.remove("d-none");
}

function clearStatus() {
  statusAlert.classList.add("d-none");
}

function setProgress(percent, stage) {
  const value = Math.max(0, Math.min(100, Number(percent || 0)));
  progressBar.style.width = `${value}%`;
  progressText.textContent = `${Math.round(value)}%`;
  progressStage.textContent = stage || "Running...";
}

function moneyOrNumber(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : "-";
  }
  return value;
}

function renderMetricCards(targetId, items) {
  const target = document.getElementById(targetId);
  if (!target) return;
  target.innerHTML = items
    .map(
      ({ label, value, col = "col-md-4" }) => `
        <div class="${col}">
          <div class="metric-card">
            <div class="metric-label">${label}</div>
            <div class="metric-value">${value}</div>
          </div>
        </div>
      `
    )
    .join("");
}

function renderSummary(summary) {
  renderMetricCards("dataset-summary", [
    { label: "Stock", value: summary.stock, col: "col-6 col-xl-3" },
    { label: "Exchange", value: summary.exchange, col: "col-6 col-xl-3" },
    { label: "Total Rows", value: moneyOrNumber(summary.total_rows), col: "col-6 col-xl-3" },
    { label: "Start Date", value: summary.start_date, col: "col-6 col-xl-3" },
    { label: "End Date", value: summary.end_date, col: "col-6 col-xl-3" },
    { label: "Train Samples", value: moneyOrNumber(summary.train_samples), col: "col-6 col-xl-3" },
    { label: "Test Samples", value: moneyOrNumber(summary.test_samples), col: "col-6 col-xl-3" },
  ]);
}

// bbb

function renderPlot(targetId, fig) {
  const target = document.getElementById(targetId);
  if (!target) return;
  if (!fig) {
    target.innerHTML = '<div class="text-muted">No chart available.</div>';
    return;
  }

  const layout = {
    ...fig.layout,
    autosize: true,
    margin: { l: 40, r: 20, t: 70, b: 40, ...(fig.layout?.margin || {}) },
  };

  if (["prediction-comparison", "trading-strategy", "equity-curve"].includes(targetId)) {
    target.style.height = "500px";
    target.style.maxHeight = "none";
    layout.height = 500;
  } else {
    const containerHeight = Math.max(420, target.clientHeight || 480);
    layout.height = containerHeight;
  }
  if (["prediction-comparison", "trading-strategy"].includes(targetId)) {
    layout.legend = {
      ...layout.legend,
      orientation: "h",
      xanchor: "left",
      x: 0,
      y: 1.02,
      yanchor: "bottom"
    };
  }
  const showModeBar = ["prediction-comparison", "trading-strategy"].includes(targetId);
  const config = { responsive: true, displayModeBar: showModeBar };
  if (showModeBar) {
    config.modeBarButtonsToRemove = ["select2d", "lasso2d"];
  }
  Plotly.react(target, fig.data, layout, config);
}

let currentJobId = null;

async function saveModelFromRun(modelName, btnEl) {
  if (!currentJobId) {
    alert("No active run found to save from.");
    return;
  }

  btnEl.disabled = true;
  btnEl.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span> Saving...';

  try {
    const response = await fetch("/api/models/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: currentJobId,
        model_name: modelName
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Failed to save model.");
    }

    const data = await response.json();
    btnEl.className = "btn btn-sm btn-success text-xs font-semibold px-2 py-1";
    btnEl.innerHTML = `✓ Saved (${data.model_id.substring(0, 6)})`;
    setStatus(`✓ Model saved successfully! Model: ${data.name} | Model ID: ${data.model_id}`, "success");
  } catch (error) {
    btnEl.disabled = false;
    btnEl.innerHTML = "Save Model";
    alert("Error saving model: " + error.message);
  }
}

function renderTable(targetId, rows, highlightBest = false) {
  const target = document.getElementById(targetId);
  if (!target) return;
  if (!rows || rows.length === 0) {
    target.innerHTML = '<div class="text-muted">No data available.</div>';
    return;
  }

  const isModelPerf = targetId === "model-performance";
  const headers = Object.keys(rows[0]);
  const table = document.createElement("table");
  table.className = "table table-sm table-hover align-middle mb-0";
  table.innerHTML = `
    <thead>
      <tr>
        ${headers.map((header) => `<th>${header}</th>`).join("")}
        ${isModelPerf ? '<th>Actions</th>' : ''}
      </tr>
    </thead>
    <tbody>
      ${rows
        .map(
          (row, index) => `
        <tr class="${highlightBest && index === 0 ? "best-model" : ""}">
          ${headers.map((header) => `<td>${row[header] ?? ""}</td>`).join("")}
          ${isModelPerf ? `
            <td>
              <button onclick="saveModelFromRun('${row.Model}', this)" class="btn btn-sm bg-indigo-600 text-white hover:bg-indigo-700 font-medium px-2.5 py-1 text-xs rounded-lg inline-flex items-center gap-1">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path><polyline points="17 21 17 13 7 13 7 21"></polyline><polyline points="7 3 7 8 15 8"></polyline></svg>
                Save Model
              </button>
            </td>
          ` : ''}
        </tr>
      `
        )
        .join("")}
    </tbody>
  `;

  target.innerHTML = "";
  const shell = document.createElement("div");
  shell.className = "table-scroll";
  shell.appendChild(table);
  target.appendChild(shell);
}

function renderStrategyPerformance(metrics) {
  renderMetricCards("strategy-performance", [
    { label: "Return", value: `${moneyOrNumber(metrics.strategy_total_return_pct)}%`, col: "col-6" },
    { label: "Buy & Hold", value: `${moneyOrNumber(metrics.buy_hold_return_pct)}%`, col: "col-6" },
    { label: "Outperformance", value: `${moneyOrNumber(metrics.outperformance_pct)}%`, col: "col-6" },
    { label: "Win Rate", value: `${moneyOrNumber(metrics.win_rate_pct)}%`, col: "col-6" },
    { label: "Sharpe Ratio", value: moneyOrNumber(metrics.sharpe_ratio), col: "col-6" },
    { label: "Max Drawdown", value: `${moneyOrNumber(metrics.max_drawdown_pct)}%`, col: "col-6" },
    { label: "Total Trades", value: moneyOrNumber(metrics.total_trades), col: "col-6" },
    { label: "Winning Trades", value: moneyOrNumber(metrics.winning_trades), col: "col-6" },
    { label: "Losing Trades", value: moneyOrNumber(metrics.losing_trades), col: "col-6" },
  ]);
}

function renderWalkForward(tableRows, fig) {
  renderTable("walk-forward-table", tableRows);
  renderPlot("walk-forward-plot", fig);
}

function renderFuture(fig, rows) {
  renderPlot("future-prediction", fig);
  renderTable("future-table", rows);
}

function renderResults(result) {
  lastResult = result;
  renderSummary(result.dataset_summary);
  renderPlot("prediction-comparison", result.comparison.figure);
  renderTable("model-performance", result.comparison.table, true);
  renderPlot("trading-strategy", result.strategy.figure);
  renderStrategyPerformance(result.strategy.metrics);
  renderPlot("equity-curve", result.strategy.equity_figure);
  renderWalkForward(result.walk_forward.table, result.walk_forward.figure);
  renderFuture(result.future_prediction.figure, result.future_prediction.table);
  resultsRoot.classList.remove("d-none");
  resizePlots();
}

async function safeFetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch (err) {
    throw new Error(`Server returned invalid response (${response.status}): ${text || response.statusText}`);
  }

  if (!response.ok) {
    const errorMsg = data?.error || data?.detail || data?.message || `Server error (${response.status})`;
    throw new Error(errorMsg);
  }

  return data;
}

async function pollJob(jobId) {
  while (true) {
    let payload;
    try {
      payload = await safeFetchJson(`/api/runs/${jobId}`);
    } catch (err) {
      setStatus(`Run query error: ${err.message}`, "danger");
      return;
    }

    setProgress(payload.progress ?? 0, payload.stage || "Running...");

    if (payload.status === "complete") {
      setStatus("Analysis complete.", "success");
      renderResults(payload.result);
      return;
    }

    if (payload.status === "error" || payload.status === "failed") {
      const errMsg = payload.error || payload.detail || "Analysis failed.";
      setStatus(errMsg, "danger");
      return;
    }

    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

function formPayload() {
  const data = new FormData(form);
  const payload = {};

  for (const [key, value] of data.entries()) {
    if (key === "selected_models") {
      payload.selected_models = payload.selected_models || [];
      payload.selected_models.push(value);
    } else if (key === "random_seed") {
      payload.random_seed = value === "" ? null : Number(value);
    } else if (["dataset_length", "time_step", "ema_period", "forecast_days", "walk_forward_days"].includes(key)) {
      payload[key] = Number(value);
    } else if (key === "train_split") {
      payload[key] = Number(value);
    } else {
      payload[key] = value;
    }
  }

  payload.selected_models = payload.selected_models || [];
  return payload;
}

function downloadBlob(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function rowsToCsv(rows) {
  if (!rows || rows.length === 0) return "";
  const headers = Object.keys(rows[0]);
  const esc = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  return [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => esc(row[header])).join(",")),
  ].join("\n");
}

function buildReportHtml(result) {
  const metrics = result.strategy.metrics || {};
  const rows = result.comparison.table || [];
  const tableHtml = rows.length
    ? `
      <table>
        <thead><tr>${Object.keys(rows[0]).map((header) => `<th>${header}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows
            .map((row) => `<tr>${Object.keys(rows[0]).map((header) => `<td>${row[header] ?? ""}</td>`).join("")}</tr>`)
            .join("")}
        </tbody>
      </table>
    `
    : "<p>No model performance data available.</p>";

  return `<!doctype html>
  <html>
  <head>
    <meta charset="utf-8" />
    <title>Trading Report</title>
    <style>
      body { font-family: Arial, sans-serif; padding: 24px; color: #111827; }
      .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 20px 0; }
      .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; }
      .label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .08em; }
      .value { font-size: 18px; font-weight: 700; margin-top: 4px; }
      table { border-collapse: collapse; width: 100%; margin-top: 12px; }
      th, td { border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }
      th { background: #f3f4f6; }
    </style>
  </head>
  <body>
    <h1>Trading Report</h1>
    <p><strong>Best Model:</strong> ${result.best_model?.name || "-"}</p>
    <div class="grid">
      <div class="card"><div class="label">Strategy Return</div><div class="value">${moneyOrNumber(metrics.strategy_total_return_pct)}%</div></div>
      <div class="card"><div class="label">Buy & Hold Return</div><div class="value">${moneyOrNumber(metrics.buy_hold_return_pct)}%</div></div>
      <div class="card"><div class="label">Sharpe Ratio</div><div class="value">${moneyOrNumber(metrics.sharpe_ratio)}</div></div>
      <div class="card"><div class="label">Max Drawdown</div><div class="value">${moneyOrNumber(metrics.max_drawdown_pct)}%</div></div>
    </div>
    <h2>Model Performance</h2>
    ${tableHtml}
  </body>
  </html>`;
}

function downloadExcel() {
  if (!lastResult || typeof XLSX === "undefined") return;
  const workbook = XLSX.utils.book_new();
  const sheets = [
    ["Dataset Summary", [lastResult.dataset_summary]],
    ["Model Performance", lastResult.comparison.table || []],
    ["Best Model", [lastResult.best_model || {}]],
    ["Strategy Performance", [lastResult.strategy.metrics || {}]],
    ["Walk Forward", lastResult.walk_forward.table || []],
    ["Future Prediction", lastResult.future_prediction.table || []],
  ];

  sheets.forEach(([name, rows]) => {
    const worksheet = XLSX.utils.json_to_sheet(rows);
    XLSX.utils.book_append_sheet(workbook, worksheet, name.slice(0, 31));
  });

  XLSX.writeFile(workbook, "trading_dashboard.xlsx");
}

function resizePlots() {
  requestAnimationFrame(() => {
    document.querySelectorAll(".js-plotly-plot").forEach((plot) => {
      if (window.Plotly && plot) Plotly.Plots.resize(plot);
    });
  });
}

selectAllBtn.addEventListener("click", () => {
  document.querySelectorAll('input[name="selected_models"]').forEach((checkbox) => {
    checkbox.checked = true;
  });
});

clearAllBtn.addEventListener("click", () => {
  document.querySelectorAll('input[name="selected_models"]').forEach((checkbox) => {
    checkbox.checked = false;
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearStatus();
  setProgress(0, "Submitting run...");
  resultsRoot.classList.add("d-none");

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text);
    }

    const payload = await response.json();
    currentJobId = payload.job_id;
    setStatus("Run started. Working through the notebook pipeline...", "primary");
    await pollJob(payload.job_id);
  } catch (error) {
    setStatus(error.message || "Unable to start run.", "danger");
  }
});

downloadCsvBtn.addEventListener("click", () => {
  if (!lastResult) return;
  downloadBlob("model_performance.csv", rowsToCsv(lastResult.comparison.table || []), "text/csv;charset=utf-8");
});

downloadExcelBtn.addEventListener("click", () => {
  downloadExcel();
});

downloadReportBtn.addEventListener("click", () => {
  if (!lastResult) return;
  downloadBlob("trading_report.html", buildReportHtml(lastResult), "text/html;charset=utf-8");
});

window.addEventListener("resize", resizePlots);
setProgress(0, "Waiting for a run.");
