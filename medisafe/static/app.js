/* MEDISAFE-GNN dashboard logic */
const selected = [];
const MAX_DRUGS = 4;

const searchInput = document.getElementById("drug-search");
const suggestions = document.getElementById("suggestions");
const chips = document.getElementById("selected");
const analyzeBtn = document.getElementById("analyze-btn");
const resultsBox = document.getElementById("results");

async function fetchDrugs(q) {
  const res = await fetch(`/api/drugs?q=${encodeURIComponent(q)}`);
  const data = await res.json();
  return data.drugs || [];
}

function renderChips() {
  chips.innerHTML = "";
  selected.forEach((name) => {
    const el = document.createElement("span");
    el.className = "chip";
    el.innerHTML = `${name} <button title="remove">&times;</button>`;
    el.querySelector("button").onclick = () => {
      selected.splice(selected.indexOf(name), 1);
      renderChips();
    };
    chips.appendChild(el);
  });
  analyzeBtn.disabled = selected.length < 2;
}

searchInput.addEventListener("input", async (e) => {
  const q = e.target.value.trim();
  if (q.length < 1) { suggestions.classList.add("hidden"); return; }
  const drugs = await fetchDrugs(q);
  suggestions.innerHTML = "";
  drugs.filter((d) => !selected.includes(d.name)).forEach((d) => {
    const li = document.createElement("li");
    li.textContent = d.name;
    li.onclick = () => {
      if (selected.length >= MAX_DRUGS) return;
      selected.push(d.name);
      searchInput.value = "";
      suggestions.classList.add("hidden");
      renderChips();
    };
    suggestions.appendChild(li);
  });
  suggestions.classList.toggle("hidden", suggestions.children.length === 0);
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-row")) suggestions.classList.add("hidden");
});

// pressing Enter adds the top suggestion as a chip (or runs analysis if full)
searchInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  e.preventDefault();
  const first = suggestions.querySelector("li");
  if (first && !suggestions.classList.contains("hidden")) {
    first.click();
  } else if (searchInput.value.trim() && selected.length < MAX_DRUGS) {
    const typed = searchInput.value.trim();
    fetchDrugs(typed).then((drugs) => {
      const exact = drugs.find((d) => d.name.toLowerCase() === typed.toLowerCase());
      if (exact) {
        selected.push(exact.name);
        searchInput.value = "";
        renderChips();
      }
    });
  }
});


function riskClass(p, type) {
  if (type === "contraindicated" || p >= 0.85) return "risk-contraindicated";
  if (type === "major" || p >= 0.7) return "risk-major";
  if (type === "moderate" || p >= 0.5) return "risk-moderate";
  return "risk-minor";
}

function badgeText(pair) {
  if (!pair.interaction_detected) return "No interaction predicted";
  const t = pair.interaction_type;
  return `Potential ${t} interaction`;
}

function renderPair(pair) {
  const p = pair.interaction_probability;
  const cls = riskClass(p, pair.interaction_detected ? pair.interaction_type : "none");
  const card = document.createElement("div");
  card.className = `pair-card ${cls}`;

  const effects = pair.adverse_effects
    .map((e) => e.source === "documented"
      ? `<span class="effect-tag documented">&#128218; ${e.effect} &middot; documented</span>`
      : `<span class="effect-tag">${e.effect} &middot; ${(e.probability * 100).toFixed(0)}%</span>`)
    .join("");

  let explanationItems = "";
  (pair.explanation.pharmacology || []).forEach((x) => {
    explanationItems += `<li>${x}</li>`;
  });
  (pair.explanation.attention_paths || []).forEach((x) => {
    explanationItems += `<li>Model attention (2-hop graph message path via
      <strong>${x.route.slice(1, -1).join(", ") || "direct edge"}</strong>):
      attention mass ${x.attention_mass}</li>`;
  });

  const known = pair.known_record
    ? `<span class="known-tag">&#128218; Documented in knowledge base (${pair.known_record.interaction_type}) &mdash; mechanism: ${pair.known_record.mechanism}</span>`
    : "";

  card.innerHTML = `
    <div class="pair-head">
      <h3>${pair.drug_a} + ${pair.drug_b}</h3>
      <span class="risk-badge ${cls}">${badgeText(pair)}</span>
    </div>
    <div class="prob-bar"><div class="prob-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
    <div class="prob-label">Predicted interaction probability: ${(p * 100).toFixed(1)}%
      &nbsp;&middot;&nbsp; confidence score: ${(pair.confidence * 100).toFixed(0)}%</div>
    ${effects ? `<div class="effects"><h4>Predicted adverse effects</h4>${effects}</div>` : ""}
    ${known}
    ${explanationItems ? `<div class="explain"><h4>Supporting evidence</h4><ul>${explanationItems}</ul></div>` : ""}
    <div class="disclaimer-inline">${pair.disclaimer}</div>
  `;
  return card;
}

analyzeBtn.addEventListener("click", async () => {
  if (selected.length < 2) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Running GNN inference\u2026";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drugs: selected }),
    });
    const data = await res.json();
    if (data.error) {
      resultsBox.classList.remove("hidden");
      resultsBox.innerHTML = `<div class="pair-card risk-major">${data.error}</div>`;
      return;
    }
    resultsBox.classList.remove("hidden");
    resultsBox.innerHTML = `
      <div class="summary-card">
        <h3>Analysis of ${data.drugs.join(" + ")}</h3>
        <span class="risk-badge ${riskClass(data.overall_risk, "major")}">
          Highest pair probability: ${(data.overall_risk * 100).toFixed(1)}%</span>
      </div>`;
    data.pairs.forEach((p) => resultsBox.appendChild(renderPair(p)));
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze combination";
  }
});

(async function init() {
  const res = await fetch("/api/model");
  const meta = await res.json();
  const m = meta.metrics || {};
  document.getElementById("model-badge").textContent =
    `GAT model \u2022 test AUC ${(m.test_auc ?? 0).toFixed(3)} \u2022 ` +
    `${meta.n_drugs} drugs \u2022 ${meta.n_known_interactions} documented interactions`;
})();
