/**
 * script.js – TruthLens: AI Fake News Detector (v4 — Multi-page SPA)
 *
 * Handles:
 *   - Client-side page routing (show/hide pages)
 *   - Navbar scroll effect & mobile hamburger
 *   - Scroll-reveal animations
 *   - News detector API calls with result display
 *   - Sample article pre-fill
 */

const API_URL = "/predict";

/* ══════════════════════════════════════════════════════════════════════════
   PAGE ROUTING
   ══════════════════════════════════════════════════════════════════════════ */
const pages = document.querySelectorAll(".page");

function navigateTo(pageName) {
    // Hide all pages
    pages.forEach((p) => {
        p.classList.remove("active");
    });

    // Show target page
    const target = document.getElementById(`page-${pageName}`);
    if (target) {
        target.classList.add("active");
    }

    // Update nav links
    document.querySelectorAll(".nav-link").forEach((l) => {
        l.classList.remove("active");
        if (l.dataset.page === pageName) l.classList.add("active");
    });

    // Scroll to top
    window.scrollTo({ top: 0, behavior: "instant" });

    // Re-trigger reveal animations for new page
    setTimeout(() => initRevealForPage(target), 50);

    // Close mobile menu
    hamburger.classList.remove("open");
    navLinks.classList.remove("open");
}

// Handle all data-page clicks (nav links, buttons, footer links)
document.addEventListener("click", (e) => {
    const trigger = e.target.closest("[data-page]");
    if (trigger) {
        e.preventDefault();
        const page = trigger.dataset.page;
        navigateTo(page);
    }
});

/* ══════════════════════════════════════════════════════════════════════════
   NAVBAR
   ══════════════════════════════════════════════════════════════════════════ */
const navbar    = document.getElementById("navbar");
const hamburger = document.getElementById("hamburger");
const navLinks  = document.getElementById("nav-links");

// Shrink navbar on scroll
window.addEventListener("scroll", () => {
    navbar.classList.toggle("scrolled", window.scrollY > 40);
});

// Hamburger toggle
hamburger.addEventListener("click", () => {
    hamburger.classList.toggle("open");
    navLinks.classList.toggle("open");
});

/* ══════════════════════════════════════════════════════════════════════════
   SCROLL REVEAL
   ══════════════════════════════════════════════════════════════════════════ */
function initRevealForPage(container) {
    if (!container) return;
    const targets = container.querySelectorAll(
        ".fact-card, .step-card, .about-card, .tips-card, .pipeline, " +
        ".section-header, .detector-input-card, .feature-card, " +
        ".samples-card, .timeline-card, .tech-details"
    );
    targets.forEach((el) => {
        el.classList.remove("visible");
        el.classList.add("reveal");
    });

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("visible");
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
    );
    targets.forEach((el) => observer.observe(el));

    // Stagger grid children
    const grids = container.querySelectorAll(
        ".facts-grid, .steps-grid, .about-grid, .tips-grid, .features-grid, .samples-grid"
    );
    grids.forEach((grid) => {
        Array.from(grid.children).forEach((child, i) => {
            child.style.transitionDelay = `${i * 0.08}s`;
        });
    });
}

// Init reveal for the default active page
initRevealForPage(document.querySelector(".page.active"));

/* ══════════════════════════════════════════════════════════════════════════
   DETECTOR
   ══════════════════════════════════════════════════════════════════════════ */
const textarea           = document.getElementById("news-input");
const checkBtn           = document.getElementById("check-btn");
const resultCard         = document.getElementById("result-card");
const resultBadge        = document.getElementById("result-badge");
const resultLabel        = document.getElementById("result-label");
const confidenceText     = document.getElementById("confidence-text");
const confidenceFill     = document.getElementById("confidence-fill");
const explanationSection = document.getElementById("explanation-section");
const explanationDesc    = document.getElementById("explanation-desc");
const keywordList        = document.getElementById("keyword-list");

checkBtn.addEventListener("click", async () => {
    const text = textarea.value.trim();

    if (!text) {
        shakeButton();
        return;
    }

    checkBtn.classList.add("loading");
    resultCard.classList.add("hidden");

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) throw new Error(`Status ${response.status}`);

        const data = await response.json();
        showResult(data);
    } catch (err) {
        console.error("Prediction error:", err);
        showError(
            "Could not reach the server. Make sure the backend is running on port 5000."
        );
    } finally {
        checkBtn.classList.remove("loading");
    }
});

/* ── Sample article buttons ─────────────────────────────────────────────── */
document.querySelectorAll(".sample-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        textarea.value = btn.dataset.sample;
        textarea.focus();
        // Scroll textarea into view
        textarea.scrollIntoView({ behavior: "smooth", block: "center" });
    });
});

/**
 * Display prediction result, confidence, explanation, and keywords.
 */
function showResult({ prediction, confidence, explanation, keywords }) {
    const isReal = prediction === "Real";
    const cls    = isReal ? "real" : "fake";
    const icon   = isReal ? "✅" : "❌";

    // Badge
    resultBadge.textContent = `${icon}  ${prediction} News`;
    resultBadge.className   = `result-badge ${cls}`;

    // Sub-label
    resultLabel.textContent = isReal
        ? "This article appears to be legitimate."
        : "This article appears to be fabricated.";

    // Confidence bar
    const pct = confidence != null ? (confidence * 100).toFixed(1) : "—";
    confidenceText.textContent = `Confidence: ${pct}%`;
    confidenceFill.className   = `confidence-fill ${cls}`;

    confidenceFill.style.width = "0%";
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            confidenceFill.style.width =
                confidence != null ? `${pct}%` : "0%";
        });
    });

    // Explanation + keywords
    if (explanation || (keywords && keywords.length > 0)) {
        explanationDesc.textContent = explanation || "";
        keywordList.innerHTML = "";
        if (keywords && keywords.length > 0) {
            keywords.forEach((word, i) => {
                const li = document.createElement("li");
                li.className = `keyword-chip ${cls}`;
                li.textContent = word;
                li.style.animationDelay = `${i * 0.06}s`;
                keywordList.appendChild(li);
            });
        }
        explanationSection.classList.remove("hidden");
    } else {
        explanationSection.classList.add("hidden");
    }

    // Reveal result card
    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/**
 * Display an error message in the result card.
 */
function showError(message) {
    resultBadge.textContent    = "⚠️  Error";
    resultBadge.className      = "result-badge fake";
    resultLabel.textContent    = message;
    confidenceText.textContent = "";
    confidenceFill.style.width = "0%";
    explanationSection.classList.add("hidden");
    resultCard.classList.remove("hidden");
}

/**
 * Quick shake animation when input is empty.
 */
function shakeButton() {
    checkBtn.style.animation = "none";
    void checkBtn.offsetWidth;
    checkBtn.style.animation = "shake 0.4s ease";
}

/* Inject shake keyframes */
const shakeStyle = document.createElement("style");
shakeStyle.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20%      { transform: translateX(-6px); }
        40%      { transform: translateX(6px); }
        60%      { transform: translateX(-4px); }
        80%      { transform: translateX(4px); }
    }
`;
document.head.appendChild(shakeStyle);
