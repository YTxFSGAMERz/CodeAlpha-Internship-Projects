(() => {
  "use strict";

  const log = document.getElementById("log");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const send = document.getElementById("send");
  const starters = JSON.parse(document.getElementById("starters").textContent || "[]");

  let busy = false;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function scrollToEnd() {
    log.scrollTop = log.scrollHeight;
  }

  function addUser(text) {
    const wrap = el("div", "msg user");
    wrap.appendChild(el("p", "bubble", text));
    log.appendChild(wrap);
    scrollToEnd();
  }

  function addTyping() {
    const wrap = el("div", "msg bot");
    const bubble = el("div", "bubble typing");
    bubble.setAttribute("aria-label", "The assistant is typing");
    for (let i = 0; i < 3; i += 1) bubble.appendChild(el("span"));
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    scrollToEnd();
    return wrap;
  }

  function buildChips(questions) {
    const row = el("div", "chips");
    questions.forEach((question) => {
      const chip = el("button", "chip", question);
      chip.type = "button";
      chip.addEventListener("click", () => ask(question));
      row.appendChild(chip);
    });
    return row;
  }

  // reply = { answer, source, score, matched_question, suggestions }
  function addBot(reply, isError = false) {
    const wrap = el("div", "msg bot");
    wrap.appendChild(el("p", isError ? "bubble error" : "bubble", reply.answer));

    if (reply.source === "faq") {
      const match = el("div", "match");
      match.appendChild(el("span", "match-question", `Matched: ${reply.matched_question}`));
      match.appendChild(el("span", "match-score", `${Math.round(reply.score * 100)}% match`));
      wrap.appendChild(match);
    }

    if (reply.suggestions && reply.suggestions.length) {
      if (reply.source === "fallback") {
        wrap.appendChild(el("p", "chips-label", "You could try one of these:"));
      }
      wrap.appendChild(buildChips(reply.suggestions));
    }

    log.appendChild(wrap);
    scrollToEnd();
  }

  async function ask(text) {
    const message = text.trim();
    if (!message || busy) return;

    busy = true;
    send.disabled = true;
    addUser(message);
    input.value = "";
    const typing = addTyping();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await response.json().catch(() => ({}));
      typing.remove();
      if (response.ok) {
        addBot(data);
      } else {
        addBot({ answer: data.error || "Something went wrong. Please try again.", suggestions: [] }, true);
      }
    } catch (error) {
      typing.remove();
      addBot({
        answer: "I can't reach the server right now. Check that the app is still running and try again.",
        suggestions: [],
      }, true);
    } finally {
      busy = false;
      send.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    ask(input.value);
  });

  addBot({
    answer: "Hi! I can answer questions about orders, delivery, returns, payments and your account. Try one of these:",
    source: "intro",
    suggestions: starters,
  });
})();
