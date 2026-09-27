/**
 * HEART ATTACK ANALYSIS — Interactive Dashboard Controller
 * Subject: Data Science Using Python
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSliders();
  initSegmentedControls();
  initPresets();
  initImageModal();
  loadOverviewData();
  initPredictionTrigger();

  // Run initial prediction on page load with default values
  setTimeout(() => {
    executePrediction();
  }, 400);
});

// ---------------------------------------------------------------------------
// TAB SWITCHING
// ---------------------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  const panes = document.querySelectorAll(".tab-pane");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");

      tabs.forEach(t => t.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const activePane = document.getElementById(targetId);
      if (activePane) activePane.classList.add("active");
    });
  });
}

// ---------------------------------------------------------------------------
// SLIDERS & LIVE LABELS
// ---------------------------------------------------------------------------
function initSliders() {
  const sliders = [
    { id: "input-age", badge: "val-age", suffix: " yrs" },
    { id: "input-trestbps", badge: "val-trestbps", suffix: " mmHg" },
    { id: "input-chol", badge: "val-chol", suffix: " mg/dL" },
    { id: "input-thalach", badge: "val-thalach", suffix: " bpm" },
    { id: "input-oldpeak", badge: "val-oldpeak", suffix: "" },
    { id: "input-ca", badge: "val-ca", suffix: "" }
  ];

  sliders.forEach(s => {
    const el = document.getElementById(s.id);
    const badge = document.getElementById(s.badge);
    if (el && badge) {
      el.addEventListener("input", () => {
        badge.textContent = el.value + s.suffix;
      });
    }
  });

  // Re-run prediction on select change
  ["input-model", "input-cp", "input-thal", "input-slope", "input-restecg"].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("change", () => executePrediction());
    }
  });
}

// ---------------------------------------------------------------------------
// SEGMENTED CONTROLS (Sex, Angina, FBS)
// ---------------------------------------------------------------------------
function initSegmentedControls() {
  const groups = [
    { hiddenId: "input-sex", btns: ["btn-sex-male", "btn-sex-female"] },
    { hiddenId: "input-exang", btns: ["btn-exang-yes", "btn-exang-no"] },
    { hiddenId: "input-fbs", btns: ["btn-fbs-yes", "btn-fbs-no"] }
  ];

  groups.forEach(g => {
    const hidden = document.getElementById(g.hiddenId);
    g.btns.forEach(btnId => {
      const btn = document.getElementById(btnId);
      if (btn) {
        btn.addEventListener("click", () => {
          g.btns.forEach(id => document.getElementById(id).classList.remove("active"));
          btn.classList.add("active");
          hidden.value = btn.getAttribute("data-val");
          executePrediction();
        });
      }
    });
  });
}

// ---------------------------------------------------------------------------
// PRESET CLINICAL PROFILES
// ---------------------------------------------------------------------------
function initPresets() {
  const healthyProfile = {
    age: 38, sex: 0, cp: 2, trestbps: 115, chol: 180,
    fbs: 0, restecg: 0, thalach: 175, exang: 0, oldpeak: 0.2,
    slope: 1, ca: 0, thal: 3
  };

  const highRiskProfile = {
    age: 67, sex: 1, cp: 4, trestbps: 160, chol: 286,
    fbs: 0, restecg: 2, thalach: 108, exang: 1, oldpeak: 2.6,
    slope: 2, ca: 2, thal: 7
  };

  const moderateProfile = {
    age: 55, sex: 1, cp: 3, trestbps: 135, chol: 245,
    fbs: 0, restecg: 1, thalach: 145, exang: 0, oldpeak: 1.0,
    slope: 2, ca: 1, thal: 3
  };

  document.getElementById("btn-load-healthy")?.addEventListener("click", () => applyProfile(healthyProfile));
  document.getElementById("btn-load-risk")?.addEventListener("click", () => applyProfile(highRiskProfile));
  document.getElementById("btn-load-moderate")?.addEventListener("click", () => applyProfile(moderateProfile));
}

function applyProfile(p) {
  setField("input-age", "val-age", p.age, " yrs");
  setField("input-trestbps", "val-trestbps", p.trestbps, " mmHg");
  setField("input-chol", "val-chol", p.chol, " mg/dL");
  setField("input-thalach", "val-thalach", p.thalach, " bpm");
  setField("input-oldpeak", "val-oldpeak", p.oldpeak, "");
  setField("input-ca", "val-ca", p.ca, "");

  document.getElementById("input-cp").value = p.cp;
  document.getElementById("input-thal").value = p.thal;
  document.getElementById("input-slope").value = p.slope;
  document.getElementById("input-restecg").value = p.restecg;

  // Sex
  document.getElementById("input-sex").value = p.sex;
  document.getElementById("btn-sex-male").classList.toggle("active", p.sex === 1);
  document.getElementById("btn-sex-female").classList.toggle("active", p.sex === 0);

  // Exang
  document.getElementById("input-exang").value = p.exang;
  document.getElementById("btn-exang-yes").classList.toggle("active", p.exang === 1);
  document.getElementById("btn-exang-no").classList.toggle("active", p.exang === 0);

  // FBS
  document.getElementById("input-fbs").value = p.fbs;
  document.getElementById("btn-fbs-yes").classList.toggle("active", p.fbs === 1);
  document.getElementById("btn-fbs-no").classList.toggle("active", p.fbs === 0);

  executePrediction();
}

function setField(id, badgeId, val, suffix) {
  const el = document.getElementById(id);
  const badge = document.getElementById(badgeId);
  if (el) el.value = val;
  if (badge) badge.textContent = val + suffix;
}

// ---------------------------------------------------------------------------
// PREDICTION TRIGGER & API CALL
// ---------------------------------------------------------------------------
function initPredictionTrigger() {
  document.getElementById("btn-run-prediction")?.addEventListener("click", () => {
    executePrediction();
  });
}

function getFormData() {
  return {
    model: document.getElementById("input-model").value,
    age: parseFloat(document.getElementById("input-age").value),
    sex: parseInt(document.getElementById("input-sex").value),
    cp: parseInt(document.getElementById("input-cp").value),
    trestbps: parseFloat(document.getElementById("input-trestbps").value),
    chol: parseFloat(document.getElementById("input-chol").value),
    fbs: parseInt(document.getElementById("input-fbs").value),
    restecg: parseInt(document.getElementById("input-restecg").value),
    thalach: parseFloat(document.getElementById("input-thalach").value),
    exang: parseInt(document.getElementById("input-exang").value),
    oldpeak: parseFloat(document.getElementById("input-oldpeak").value),
    slope: parseInt(document.getElementById("input-slope").value),
    ca: parseInt(document.getElementById("input-ca").value),
    thal: parseInt(document.getElementById("input-thal").value)
  };
}

async function executePrediction() {
  const payload = getFormData();
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      renderPredictionResult(data);
    }
  } catch (err) {
    console.error("Prediction error:", err);
  }
}

function renderPredictionResult(data) {
  const percentEl = document.getElementById("risk-percentage");
  const badgeEl = document.getElementById("risk-badge");
  const adviceEl = document.getElementById("risk-advice");
  const meterFill = document.getElementById("meter-fill");

  // Animate percentage
  const prob = data.probability;
  percentEl.textContent = prob.toFixed(1) + "%";

  // Meter arc calculation (total dasharray = 251.2 for 180 deg arc)
  const totalLength = 251.2;
  const offset = totalLength - (prob / 100) * totalLength;
  meterFill.style.strokeDashoffset = offset;

  // Color coding
  badgeEl.className = "status-badge " + data.badge_class;
  badgeEl.textContent = `${data.status} • ${data.risk_tier.toUpperCase()}`;

  if (prob < 30) {
    meterFill.style.stroke = "#10b981"; // Emerald
  } else if (prob < 65) {
    meterFill.style.stroke = "#f59e0b"; // Amber
  } else {
    meterFill.style.stroke = "#f43f5e"; // Rose
  }

  adviceEl.textContent = data.clinical_advice;

  // Update vitals strip
  const s = data.patient_summary;
  document.getElementById("vital-age-sex").textContent = `${s.age} / ${s.sex}`;
  document.getElementById("vital-bp").textContent = s.trestbps;
  document.getElementById("vital-chol").textContent = s.chol;
  document.getElementById("vital-hr").textContent = s.thalach;
}

// ---------------------------------------------------------------------------
// LOAD DATASET OVERVIEW & MODEL BENCHMARKS
// ---------------------------------------------------------------------------
async function loadOverviewData() {
  try {
    const res = await fetch("/api/overview");
    const data = await res.json();

    // 1. KPI cards
    document.getElementById("kpi-dataset-records").textContent = `${data.dataset.total_records} Patients`;
    if (data.metrics["Random Forest"]) {
      document.getElementById("kpi-accuracy").textContent = `${data.metrics["Random Forest"].accuracy}%`;
      document.getElementById("kpi-recall").textContent = `${data.metrics["Random Forest"].recall}%`;
      document.getElementById("kpi-auc").textContent = `${data.metrics["Random Forest"].roc_auc}%`;
    }

    // 2. Feature Importance Mini List
    renderFeatureImportance(data.feature_importance);

    // 3. Model Benchmark Cards & Table
    renderModelBenchmarks(data.metrics);

    // 4. Confusion Matrices
    renderConfusionMatrices(data.confusion_matrices);
  } catch (err) {
    console.error("Failed to load overview data:", err);
  }
}

function renderFeatureImportance(fiList) {
  const container = document.getElementById("feature-importance-list");
  if (!container) return;

  const top4 = fiList.slice(0, 4);
  const friendlyNames = {
    thal: "Thalassemia Perfusion Scan",
    ca: "Number of Blocked Vessels",
    oldpeak: "ST Depression on Exercise",
    thalach: "Max Heart Rate Capacity",
    cp: "Chest Pain Category",
    age: "Patient Age"
  };

  container.innerHTML = top4.map(item => `
    <div class="fi-item">
      <div class="fi-meta">
        <span class="fi-name">${friendlyNames[item.feature] || item.feature}</span>
        <span class="fi-score">${item.importance}%</span>
      </div>
      <div class="fi-bar-track">
        <div class="fi-bar-fill" style="width: ${item.importance * 3.5}%"></div>
      </div>
    </div>
  `).join("");
}

function renderModelBenchmarks(metrics) {
  const container = document.getElementById("models-benchmark-container");
  const tableBody = document.getElementById("benchmark-table-body");
  if (!container || !tableBody) return;

  const tags = {
    "Random Forest": { tag: "Ensemble Top Performer", cls: "tag-top", isFeatured: true, useCase: "Primary clinical risk screening" },
    "Logistic Regression": { tag: "Linear Baseline", cls: "tag-linear", isFeatured: false, useCase: "High interpretability & probability odds" },
    "K-Nearest Neighbors": { tag: "Distance-Based", cls: "tag-knn", isFeatured: false, useCase: "Instance similarity & 100% recall" },
    "Decision Tree": { tag: "Rule-Based", cls: "tag-tree", isFeatured: false, useCase: "Simple IF-THEN clinical flowcharts" }
  };

  // Cards
  container.innerHTML = Object.entries(metrics).map(([name, m]) => {
    const meta = tags[name] || { tag: "Algorithm", cls: "tag-linear", isFeatured: false };
    return `
      <div class="model-card ${meta.isFeatured ? 'featured' : ''}">
        <div class="model-card-header">
          <h3 class="model-name">${name}</h3>
          <span class="model-tag ${meta.cls}">${meta.tag}</span>
        </div>
        <div class="model-metrics-list">
          <div class="metric-row">
            <span class="metric-label">Accuracy:</span>
            <span class="metric-val">${m.accuracy}%</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Recall (Sensitivity):</span>
            <span class="metric-val highlight-cyan">${m.recall}%</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Precision:</span>
            <span class="metric-val">${m.precision}%</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">F1-Score:</span>
            <span class="metric-val">${m.f1_score}%</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">ROC-AUC:</span>
            <span class="metric-val highlight-purple">${m.roc_auc}%</span>
          </div>
        </div>
      </div>
    `;
  }).join("");

  // Table
  tableBody.innerHTML = Object.entries(metrics).map(([name, m]) => {
    const meta = tags[name] || { useCase: "Classification" };
    return `
      <tr>
        <td><strong>${name}</strong></td>
        <td class="table-num">${m.accuracy}%</td>
        <td class="table-num">${m.precision}%</td>
        <td class="table-num highlight-cyan"><strong>${m.recall}%</strong></td>
        <td class="table-num">${m.f1_score}%</td>
        <td class="table-num highlight-purple">${m.roc_auc}%</td>
        <td>${meta.useCase}</td>
      </tr>
    `;
  }).join("");
}

function renderConfusionMatrices(cmMap) {
  const container = document.getElementById("cm-grid-container");
  if (!container) return;

  container.innerHTML = Object.entries(cmMap).map(([name, cm]) => `
    <div class="cm-card">
      <div class="cm-card-title">${name}</div>
      <div class="cm-matrix-2x2">
        <div class="cm-cell cm-cell-tn">
          <div class="cm-val">${cm.tn}</div>
          <div class="cm-lbl">True Negatives (TN)</div>
        </div>
        <div class="cm-cell cm-cell-fp">
          <div class="cm-val">${cm.fp}</div>
          <div class="cm-lbl">False Positives (FP)</div>
        </div>
        <div class="cm-cell cm-cell-fn">
          <div class="cm-val">${cm.fn}</div>
          <div class="cm-lbl">False Negatives (FN)</div>
        </div>
        <div class="cm-cell cm-cell-tp">
          <div class="cm-val">${cm.tp}</div>
          <div class="cm-lbl">True Positives (TP)</div>
        </div>
      </div>
    </div>
  `).join("");
}

// ---------------------------------------------------------------------------
// LIGHTBOX MODAL
// ---------------------------------------------------------------------------
function initImageModal() {
  const modal = document.getElementById("image-modal");
  const modalImg = document.getElementById("modal-img");
  const caption = document.getElementById("modal-caption");
  const closeBtn = document.getElementById("modal-close");

  document.querySelectorAll(".gallery-img").forEach(img => {
    img.addEventListener("click", () => {
      modal.classList.add("active");
      modalImg.src = img.src;
      caption.textContent = img.alt;
    });
  });

  closeBtn?.addEventListener("click", () => modal.classList.remove("active"));
  modal?.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("active");
  });
}
