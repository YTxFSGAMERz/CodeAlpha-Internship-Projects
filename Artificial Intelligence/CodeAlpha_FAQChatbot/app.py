"""Flask web app with a chat window for the FAQ chatbot.

Run with:  python app.py   then open http://127.0.0.1:5000
"""

from flask import Flask, jsonify, render_template, request

from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot()          # loads the FAQs and builds the TF-IDF index once

MAX_MESSAGE_CHARS = 300


@app.get("/")
def index():
    return render_template(
        "index.html",
        store=bot.store,
        starters=bot.suggested_questions(),
        max_chars=MAX_MESSAGE_CHARS,
    )


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify(error="Type a question first."), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify(error=f"Keep your question under {MAX_MESSAGE_CHARS} characters."), 400
    return jsonify(bot.ask(message))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
