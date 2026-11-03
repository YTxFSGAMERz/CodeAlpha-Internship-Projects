(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const el = {
    sourceLang: $("source-lang"),
    targetLang: $("target-lang"),
    sourceText: $("source-text"),
    output: $("output"),
    translate: $("translate-btn"),
    swap: $("swap-btn"),
    copy: $("copy-btn"),
    speak: $("speak-btn"),
    clear: $("clear-btn"),
    count: $("char-count"),
    status: $("status"),
    hint: $("shortcut-hint"),
  };

  const MAX_CHARS = Number(el.sourceText.getAttribute("maxlength")) || 5000;
  // A few Google language codes are spelled differently by browsers' speech engines.
  const SPEECH_CODES = { iw: "he", jw: "jv", tl: "fil" };
  const PREFS_KEY = "translator.languages";

  let busy = false;
  let turns = 0;

  /* ---------- small helpers ---------- */

  function setStatus(message, kind = "") {
    el.status.textContent = message;
    el.status.dataset.kind = kind;
  }

  function hasOutput() {
    return el.output.dataset.empty === "false";
  }

  function showOutput(text) {
    el.output.textContent = text;
    el.output.dataset.empty = "false";
    el.copy.disabled = false;
    el.speak.disabled = false;
  }

  function clearOutput() {
    el.output.textContent = "";
    el.output.dataset.empty = "true";
    el.copy.disabled = true;
    el.speak.disabled = true;
  }

  function updateCounter() {
    el.count.textContent = `${el.sourceText.value.length.toLocaleString()} / ${MAX_CHARS.toLocaleString()}`;
  }

  function updateSwapState() {
    const auto = el.sourceLang.value === "auto";
    const label = auto ? "Choose a source language to swap languages" : "Swap languages";
    el.swap.disabled = auto;
    el.swap.title = label;
    el.swap.setAttribute("aria-label", label);
  }

  function flash(button, text) {
    const label = button.querySelector(".label");
    const original = label.textContent;
    label.textContent = text;
    setTimeout(() => { label.textContent = original; }, 1500);
  }

  /* ---------- remember the last languages ---------- */

  function readPrefs() {
    try {
      return JSON.parse(localStorage.getItem(PREFS_KEY)) || {};
    } catch (error) {
      return {};
    }
  }

  function savePrefs() {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify({
        source: el.sourceLang.value,
        target: el.targetLang.value,
      }));
    } catch (error) {
      /* storage can be blocked; the app works without it */
    }
  }

  function applyPrefs() {
    const prefs = readPrefs();
    const has = (select, value) => Array.from(select.options).some((o) => o.value === value);
    if (prefs.source && has(el.sourceLang, prefs.source)) el.sourceLang.value = prefs.source;
    if (prefs.target && has(el.targetLang, prefs.target)) el.targetLang.value = prefs.target;
  }

  /* ---------- actions ---------- */

  async function translate() {
    if (busy) return;
    const text = el.sourceText.value.trim();
    if (!text) {
      setStatus("Enter some text to translate.", "error");
      el.sourceText.focus();
      return;
    }

    busy = true;
    el.translate.disabled = true;
    el.translate.textContent = "Translating…";
    setStatus("");

    try {
      const response = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          source: el.sourceLang.value,
          target: el.targetLang.value,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Translation failed. Try again.");
      showOutput(data.translation);
    } catch (error) {
      clearOutput();
      const offline = error instanceof TypeError;
      setStatus(offline ? "Can't reach the translator server. Is it still running?" : error.message, "error");
    } finally {
      busy = false;
      el.translate.disabled = false;
      el.translate.textContent = "Translate";
    }
  }

  function swapLanguages() {
    if (el.sourceLang.value === "auto") return;
    const from = el.sourceLang.value;
    el.sourceLang.value = el.targetLang.value;
    el.targetLang.value = from;
    if (hasOutput()) {
      el.sourceText.value = el.output.textContent;
      updateCounter();
    }
    clearOutput();
    setStatus("");
    turns += 1;
    el.swap.style.setProperty("--turns", turns);
    savePrefs();
    updateSwapState();
  }

  function legacyCopy(text) {
    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.setAttribute("readonly", "");
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.appendChild(scratch);
    scratch.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(scratch);
    if (!ok) throw new Error("copy command was rejected");
  }

  async function copyOutput() {
    if (!hasOutput()) return;
    const text = el.output.textContent;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        legacyCopy(text);
      }
      flash(el.copy, "Copied");
    } catch (error) {
      setStatus("Couldn't copy automatically. Select the text and press Ctrl + C.", "error");
    }
  }

  function speakOutput() {
    if (!hasOutput()) return;
    if (!("speechSynthesis" in window)) {
      setStatus("This browser can't read text aloud.", "error");
      return;
    }
    const synth = window.speechSynthesis;
    const code = el.targetLang.value;
    const lang = SPEECH_CODES[code] || code;
    const base = lang.split("-")[0].toLowerCase();
    const voices = synth.getVoices();
    const voice =
      voices.find((v) => v.lang.replace("_", "-").toLowerCase() === lang.toLowerCase()) ||
      voices.find((v) => v.lang.split(/[-_]/)[0].toLowerCase() === base);

    if (voices.length && !voice) {
      const name = el.targetLang.options[el.targetLang.selectedIndex].text;
      setStatus(`No ${name} voice is installed in this browser, so it can't be read aloud.`, "error");
      return;
    }

    synth.cancel();
    const utterance = new SpeechSynthesisUtterance(el.output.textContent);
    utterance.lang = voice ? voice.lang : lang;
    if (voice) utterance.voice = voice;
    synth.speak(utterance);
    setStatus("");
  }

  /* ---------- wiring ---------- */

  el.translate.addEventListener("click", translate);
  el.swap.addEventListener("click", swapLanguages);
  el.copy.addEventListener("click", copyOutput);
  el.speak.addEventListener("click", speakOutput);

  el.clear.addEventListener("click", () => {
    el.sourceText.value = "";
    updateCounter();
    clearOutput();
    setStatus("");
    el.sourceText.focus();
  });

  el.sourceText.addEventListener("input", updateCounter);
  el.sourceText.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      translate();
    }
  });

  el.sourceLang.addEventListener("change", () => {
    savePrefs();
    updateSwapState();
  });

  el.targetLang.addEventListener("change", () => {
    savePrefs();
    // Switching the target after a translation re-translates straight away.
    if (hasOutput() && el.sourceText.value.trim()) translate();
  });

  applyPrefs();
  updateCounter();
  updateSwapState();
  clearOutput();
  if (/Mac|iPhone|iPad/.test(navigator.platform || "")) el.hint.textContent = "⌘ + Enter";
})();
