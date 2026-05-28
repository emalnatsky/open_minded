// =====================================================================
// Leo's Geheugen — Three-screen tablet GUI
// CRITICAL: handleUMUpdate, mapCategory, liveUM, CATEGORIES, FIELD_LABELS,
// CAT_LABEL_TO_KEY, and the Socket.IO handler are preserved exactly.
// =====================================================================

// ── CATEGORIES (DO NOT CHANGE the keys/labels) ───────────────────────
const CATEGORIES = {
  hobby:      { label: "Hobby's",  icon: "🎨", iconImg: "/static/images/icons/hobby.png",       color: "#3d8b3d" },
  sport:      { label: "Sport",    icon: "⚽", iconImg: "/static/images/icons/sports.png",      color: "#1976d2" },
  muziek:     { label: "Muziek",   icon: "🎵", iconImg: "/static/images/icons/instruments.png", color: "#7b1fa2" },
  boeken:     { label: "Boeken",   icon: "📚", iconImg: "/static/images/icons/fav_book.png",    color: "#fb8c00" },
  sociaal:    { label: "Sociaal",  icon: "👫", iconImg: "/static/images/icons/social.png",      color: "#e91e63" },
  dieren:     { label: "Dieren",   icon: "🐾", iconImg: "/static/images/icons/dieren.png",      color: "#558b2f" },
  eten:       { label: "Eten",     icon: "🍕", iconImg: "/static/images/icons/eten.png",        color: "#c62828" },
  school:     { label: "School",   icon: "📖", iconImg: "/static/images/icons/school.png",      color: "#37474f" },
  aspiratie:  { label: "Inspiratie",   icon: "⭐", iconImg: "/static/images/icons/aspiration.png",  color: "#f57f17" },
};

window.liveUM = {};
let liveUM = window.liveUM;
let currentCategory = null;
let currentScreen = "welcome";

// ── Phase-gated category locking ─────────────────────────────────────
// Set by the dialogue via session_state.json → um_update broadcast.
// Empty set = all categories locked (session not yet started).
// When the dialogue mentions a category, it's added here.
let unlockedCategories = new Set();
let memoryAccessActive = false;
let visibleFields = new Set();
let currentMistakes = {};

function shouldShowFieldInMemoryAccess(field) {
  return !memoryAccessActive || visibleFields.has(field);
}

function unresolvedMistakeForField(field) {
  if (!memoryAccessActive) return null;
  for (const mistake of Object.values(currentMistakes || {})) {
    if (!mistake || mistake.corrected) continue;
    if (mistake.field === field && mistake.wrong) return mistake;
  }
  return null;
}

function displayValueForField(field, meta) {
  const mistake = unresolvedMistakeForField(field);
  if (mistake) return mistake.wrong;
  return meta.value || "\u2014";
}

function hasDisplayValue(value) {
  return value && value !== "\u2014" && value !== "â€”";
}

function isCategoryUnlocked(catKey) {
  // If no state received yet, lock everything
  if (unlockedCategories.size === 0) return false;
  return unlockedCategories.has(catKey);
}

function applyLocking() {
  // Re-render TOC lock state without full rebuild
  document.querySelectorAll(".toc-item").forEach(li => {
    const key = li.dataset.catKey;
    if (!key) return;
    const unlocked = isCategoryUnlocked(key);
    li.classList.toggle("toc-locked", !unlocked);
    li.setAttribute("aria-disabled", unlocked ? "false" : "true");
  });
}

// ── Child identity (populated by um_update broadcasts) ───────────────
let currentChildName = "";

function setChildName(name) {
  const clean = (name || "").trim();
  if (!clean || clean === currentChildName) return;
  currentChildName = clean;

  // Update the closed-book cover on the welcome screen
  const nameEl = document.getElementById("welcomeBookName");
  if (nameEl) nameEl.textContent = clean;

  // Reveal the "van [name]" portion now that we have a real name
  const titleBlock = document.getElementById("welcomeBookTitle");
  if (titleBlock) titleBlock.classList.add("name-ready");

  // Accessibility label
  const book = document.getElementById("welcomeBook");
  if (book) book.setAttribute("aria-label", `Open ${clean}'s geheugenboek`);
}

// ── Label-to-key mapping (DO NOT CHANGE) ─────────────────────────────
const CAT_LABEL_TO_KEY = {
  "hobby's":         "hobby",
  "hobbies":         "hobby",
  "sport":           "sport",
  "muziek":          "muziek",
  "music":           "muziek",
  "boeken":          "boeken",
  "books":           "boeken",
  "vrije tijd":      "vrije_tijd",
  "vrije_tijd":      "vrije_tijd",
  "free time":       "vrije_tijd",
  "sociaal":         "sociaal",
  "social":          "sociaal",
  "dieren":          "dieren",
  "animals":         "dieren",
  "eten":            "eten",
  "food":            "eten",
  "school":          "school",
  "dromen & idolen": "aspiratie",
  "aspiratie":       "aspiratie",
  "aspiration":      "aspiratie",
  "dromen":          "aspiratie",
};

function mapCategory(rawLabel) {
  if (!rawLabel) return "overig";
  const lower = rawLabel.toLowerCase().trim().replace(/ /g, "_");
  if (CATEGORIES[lower]) return lower;
  const lower2 = rawLabel.toLowerCase().trim();
  if (CAT_LABEL_TO_KEY[lower2]) return CAT_LABEL_TO_KEY[lower2];
  for (const [key, val] of Object.entries(CAT_LABEL_TO_KEY)) {
    if (lower2.includes(key) || key.includes(lower2)) return val;
  }
  return "overig";
}

// ── Field labels (DO NOT CHANGE) ─────────────────────────────────────
const FIELD_LABELS = {
  hobby_fav:              "Lievelingshobby",
  hobbies:                "Hobbies",
  sports_enjoys:          "Houdt van sport",
  sports_fav:             "Lievelingssport",
  sports_plays:           "Doet zelf",
  sports_fav_play:        "Speelt het liefst",
  music_enjoys:           "Houdt van muziek",
  music_plays_instrument: "Speelt instrument",
  music_instrument:       "Instrument",
  books_enjoys:           "Houdt van boeken",
  books_fav_genre:        "Favoriete genre",
  books_fav_title:        "Lievelingsboek",
  freetime_fav:           "Liefste bezigheid",
  has_best_friend:        "Heeft beste vriend",
  animals_enjoys:         "Houdt van dieren",
  animal_fav:             "Lievelingsdier",
  has_pet:                "Heeft huisdier",
  pet_type:               "Soort huisdier",
  pet_name:               "Naam huisdier",
  fav_food:               "Lievelingseten",
  fav_subject:            "Lievelingsvak",
  school_strength:        "Goed in",
  school_difficulty:      "Lastig vak",
  interest:               "Interesses",
  aspiration:             "Wil later worden",
  role_model:             "Voorbeeld",
  age:                    "Leeftijd",
  name:                   "Naam",
  exposure:               "Hebben we al gepraat?",
};
function fieldLabel(key) {
  return FIELD_LABELS[key] || key.replace(/_/g, " ");
}
const HIDDEN_FIELDS = new Set([
  "hobby_talk",
  "sports_talk",
  "sports_play_talk",
  "music_talk",
  "books_talk",
  "pet_talk",
  "animal_talk",
]);

const FIELD_VALUE_LABELS = {
  exposure: {
    new:      "Het is onze eerste keer",
    returning: "We hebben al gepraat",
  },
};

function fieldValue(field, value) {
  const mapped = FIELD_VALUE_LABELS[field];
  if (mapped && mapped[value] !== undefined) return mapped[value];
  return value;
}

// =====================================================================
// SCREEN NAVIGATION
// =====================================================================
function showScreen(name) {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  const target = document.getElementById("screen-" + name);
  if (!target) return;
  target.classList.add("active");
  currentScreen = name;

  // Trigger TOC book entrance animation
  if (name === "toc") {
    const book = document.getElementById("tocBook");
    book.classList.remove("book-enter");
    void book.offsetWidth; // force reflow
    book.classList.add("book-enter");
    buildTOC();
  }
}

// =====================================================================
// SCREEN 1 — Welcome bubble change after 3s, tap book to advance
// =====================================================================
function initWelcome() {
  const text = document.getElementById("welcomeBubbleText");
  const book = document.getElementById("welcomeBook");

  // Set the bubble text once and never change it
  text.textContent = "Klik op mijn boek om meer te ontdekken!";

  // Tap or click book to go to TOC
  const goToToc = () => showScreen("toc");
  book.addEventListener("click", goToToc);
  book.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); goToToc(); }
  });
}

// =====================================================================
// SCREEN 2 — Table of Contents
// =====================================================================
function buildTOC() {
  const left  = document.getElementById("tocListLeft");
  const right = document.getElementById("tocListRight");
  left.innerHTML = "";
  right.innerHTML = "";

  const entries = Object.entries(CATEGORIES);
  entries.forEach(([key, cat], i) => {
    const li = document.createElement("li");
    li.className = "toc-item";
    li.dataset.catKey = key;
    if (!isCategoryUnlocked(key)) li.classList.add("toc-locked");
    li.style.animationDelay = (0.4 + i * 0.05) + "s";
    li.innerHTML = `
      <span class="toc-icon">${cat.iconImg ? `<img src="${cat.iconImg}" alt="" />` : cat.icon}</span>
      <span class="toc-label">${cat.label}</span>
      <span class="toc-lock-icon">🔒</span>
      <span class="toc-dots"></span>
      <span class="toc-page">p.${i + 1}</span>
    `;
    li.addEventListener("click", () => {
      if (!isCategoryUnlocked(key)) return;  // locked — ignore tap
      openCategory(key);
    });
    (i < 4 ? left : right).appendChild(li);
  });
}

// =====================================================================
// SCREEN 3 — Category page 
// =====================================================================
function openCategory(catKey) {
  const cat = CATEGORIES[catKey];
  if (!cat) return;
  currentCategory = catKey;

  // Update header
  const iconEl = document.getElementById("catIconBig");
  if (cat.iconImg) {
    iconEl.innerHTML = `<img src="${cat.iconImg}" alt="" />`;
  } else {
    iconEl.textContent = cat.icon;
  }
  document.getElementById("catTitleBig").textContent = cat.label;
  document.getElementById("catUnderline").style.background = cat.color;
  document.getElementById("catUnderline").style.boxShadow =
    `0 5px 0 0 ${shadeColor(cat.color, -25)}`;
  document.getElementById("catTitleBig").style.color = cat.color;

  // Page meta
  // const idx = Object.keys(CATEGORIES).indexOf(catKey);
  // document.getElementById("catPageNum").textContent = String(idx + 1).padStart(2, "0");

  // Bubble
  document.getElementById("catBubbleText").textContent = "Kijk wat ik weet!";

  // Render pills (initial — no eraser, just write-in)
  renderPills(catKey, /*initial*/ true);

  showScreen("category");
}

function renderPills(catKey, initial) {
  const cat   = CATEGORIES[catKey];
  const panel = document.getElementById("pillsContainer");
  const empty = document.getElementById("emptyState");
  const data  = liveUM[catKey] || {};

  const entries = Object.entries(data).filter(
    ([field, v]) => !HIDDEN_FIELDS.has(field) && shouldShowFieldInMemoryAccess(field) && v && hasDisplayValue(v.value)
  );

  document.getElementById("catCount").textContent = entries.length;

  if (entries.length === 0) {
    panel.innerHTML = "";
    empty.classList.add("show");
    return;
  }
  empty.classList.remove("show");

  // On initial open: rebuild from scratch with stagger
  if (initial) {
    panel.innerHTML = "";
    entries.forEach(([field, meta], i) => {
      const pill = buildPill(field, meta.value, cat.color);
      pill.classList.add("pill-enter");
      pill.style.animationDelay = (i * 0.07) + "s";
      panel.appendChild(pill);
    });
  }
}

function buildPill(field, value, color) {
  const dark = shadeColor(color, -30);
  const pill = document.createElement("div");
  pill.className = "pill";
  pill.dataset.field = field;
  pill.dataset.value = value;
  pill.style.setProperty("--pill-color", dark);
  pill.innerHTML = `
    <span class="pill-label">${fieldLabel(field)}</span>
    <span class="pill-value">${escapeHTML(fieldValue(field, value))}</span>
  `;
  return pill;
}

// =====================================================================
// LIVE UM UPDATES — eraser → write-in
// =====================================================================
function applyDiffToOpenCategory(catKey, changedFields) {
  if (currentScreen !== "category" || currentCategory !== catKey) return;

  const cat   = CATEGORIES[catKey];
  const panel = document.getElementById("pillsContainer");
  const data  = liveUM[catKey] || {};

  changedFields.forEach(({ field, oldValue, newValue }) => {
    if (HIDDEN_FIELDS.has(field)) return;     
    const existing = panel.querySelector(`.pill[data-field="${field}"]`);

    if (hasDisplayValue(newValue)) {
      if (existing && hasDisplayValue(oldValue) && oldValue !== newValue) {
        // ── UPDATE: erase then write ─────────────────────────────
        existing.classList.add("pill-erasing");
        setTimeout(() => {
          const newPill = buildPill(field, newValue, cat.color);
          // mark for write-in animation
          newPill.querySelector(".pill-value").classList.add("writing");
          newPill.classList.add("pill-enter");
          existing.replaceWith(newPill);
        }, 2600);  // synced with the 2.5s erase duration
      } else if (!existing) {
        // ── NEW field appearing ─────────────────────────────────
        const newPill = buildPill(field, newValue, cat.color);
        newPill.classList.add("pill-enter");
        newPill.querySelector(".pill-value").classList.add("writing");
        panel.appendChild(newPill);
        // empty-state hide
        document.getElementById("emptyState").classList.remove("show");
      } else {
        // No change in value — just refresh content (no animation)
        existing.dataset.value = newValue;
      }
    } else if (existing) {
      // Value cleared — erase
      existing.classList.add("pill-erasing");
      setTimeout(() => existing.remove(), 2600);
    }
  });

  // refresh count
  setTimeout(() => {
    const visible = panel.querySelectorAll(".pill:not(.pill-erasing)").length;
    document.getElementById("catCount").textContent = visible;
    if (visible === 0) document.getElementById("emptyState").classList.add("show");
  }, 2700);
}

// =====================================================================
// handleUMUpdate — CRITICAL: signature & socket contract unchanged
// =====================================================================
function handleUMUpdate(data) {
  console.log("handleUMUpdate called with", Object.keys(data.fields || {}).length, "fields");
  const fields    = data.fields    || {};
  const timestamp = data.timestamp || "";
  memoryAccessActive = Boolean(data.memory_access_active);
  visibleFields = new Set(Array.isArray(data.visible_fields) ? data.visible_fields : []);
  currentMistakes = data.mistakes || {};

  // Update child name on the closed book cover (Screen 1)
  if (data.child_name) setChildName(data.child_name);

  // Update phase-gated locking
  if (Array.isArray(data.unlocked_categories)) {
    unlockedCategories = new Set(data.unlocked_categories);
    applyLocking();
  }

  // Group by category & detect per-field changes vs. prior liveUM
  const grouped = {};
  const diffsByCategory = {}; // { catKey: [{field, oldValue, newValue}, ...] }

  Object.entries(fields).forEach(([field, meta]) => {
    if (!shouldShowFieldInMemoryAccess(field)) return;

    const rawCat = meta.category || "";
    const catKey = mapCategory(rawCat);

    if (!grouped[catKey]) grouped[catKey] = {};
    if (!diffsByCategory[catKey]) diffsByCategory[catKey] = [];

    const prevMeta  = liveUM[catKey] && liveUM[catKey][field];
    const prevValue = prevMeta ? prevMeta.value : null;
    const newValue  = displayValueForField(field, meta);

    grouped[catKey][field] = {
      value:   newValue,
      updated: prevValue !== null && prevValue !== newValue && hasDisplayValue(newValue),
    };

    if (prevValue !== newValue) {
      diffsByCategory[catKey].push({ field, oldValue: prevValue, newValue });
    }
  });

  // Commit to liveUM. Replace category maps so memory-access filtering can
  // hide fields that were visible in a previous tablet update.
  Object.keys(liveUM).forEach(catKey => delete liveUM[catKey]);
  Object.assign(liveUM, grouped);

  // Connection / status indicator
  setConn("connected", "Live ✓ " + (timestamp || ""));

  // If a category screen is open, apply per-field eraser/write diffs
  if (currentScreen === "category" && currentCategory) {
    if (memoryAccessActive) {
      renderPills(currentCategory, true);
      return;
    }
    const diffs = diffsByCategory[currentCategory] || [];
    if (diffs.length > 0) {
      applyDiffToOpenCategory(currentCategory, diffs);
    } else {
      // No diffs for current category, but ensure pills exist (first poll)
      const panel = document.getElementById("pillsContainer");
      if (panel.children.length === 0) renderPills(currentCategory, true);
    }
  }
}

// =====================================================================
// Connection indicator
// =====================================================================
function setConn(state, text) {
  const ind = document.getElementById("connIndicator");
  const txt = document.getElementById("connText");
  ind.classList.remove("connected", "error", "connecting");
  if (state) ind.classList.add(state);
  if (txt) txt.textContent = text;
}

// =====================================================================
// Utilities
// =====================================================================
function escapeHTML(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function shadeColor(hex, percent) {
  // negative percent → darker; positive → lighter
  const num = parseInt(hex.replace("#", ""), 16);
  const r = (num >> 16) & 0xff;
  const g = (num >>  8) & 0xff;
  const b =  num        & 0xff;
  const adj = (c) => {
    const v = Math.round(c + (percent / 100) * 255);
    return Math.max(0, Math.min(255, v));
  };
  const r2 = adj(r), g2 = adj(g), b2 = adj(b);
  return "#" + ((1 << 24) + (r2 << 16) + (g2 << 8) + b2).toString(16).slice(1);
}

// =====================================================================
// Back buttons
// =====================================================================
document.querySelectorAll(".back-btn").forEach(btn => {
  btn.addEventListener("click", () => showScreen(btn.dataset.backTo));
});

// =====================================================================
// Socket.IO — CRITICAL HANDLER, DO NOT CHANGE
// =====================================================================
try {
  window.socket = io({ transports: ["websocket", "polling"] });

  window.socket.on("connect",        () => setConn("connected", "Verbonden ✓"));
  window.socket.on("disconnect",     () => setConn("error", "Verbinding verbroken"));
  window.socket.on("connect_error",  () => setConn("error", "Server niet bereikbaar"));

  window.socket.on("sic/webinfo", (data) => {
    if (data.label === "um_update") {
      handleUMUpdate(data.message);
    }
  });
} catch (e) {
  setConn("error", "Socket.IO niet beschikbaar");
}

// =====================================================================
// Auto-scale the 1280x800 design canvas to fit any viewport
// =====================================================================
function fitToViewport() {
  const designW = 1280;
  const designH = 800;
  const scale = Math.min(
    window.innerWidth  / designW,
    window.innerHeight / designH
  );
  document.body.style.transform =
    `translate(-50%, -50%) scale(${scale})`;
}
window.addEventListener("resize", fitToViewport);
window.addEventListener("orientationchange", fitToViewport);

// =====================================================================
// Init
// =====================================================================
document.addEventListener("DOMContentLoaded", () => {
  fitToViewport();
  initWelcome();
  setConn("connecting", "Verbinden…");

  // Escape navigates back
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (currentScreen === "category") showScreen("toc");
    else if (currentScreen === "toc") showScreen("welcome");
  });
});
