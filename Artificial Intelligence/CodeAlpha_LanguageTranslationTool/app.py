"""Flask web app for the Language Translation Tool.

Run with:  python app.py   then open http://127.0.0.1:5000
"""

from flask import Flask, jsonify, render_template, request

import translator

app = Flask(__name__)

DEFAULT_TARGET = "hi"  # the language preselected on the right-hand side


@app.get("/")
def index():
    return render_template(
        "index.html",
        languages=translator.language_options(),
        max_chars=translator.MAX_CHARS,
        default_target=DEFAULT_TARGET,
    )


@app.post("/api/translate")
def api_translate():
    data = request.get_json(silent=True) or {}
    source = data.get("source") or translator.AUTO
    target = data.get("target") or DEFAULT_TARGET
    try:
        result = translator.translate(data.get("text", ""), source, target)
    except translator.TranslationError as err:
        return jsonify(error=err.message), err.status
    return jsonify(translation=result, source=source, target=target)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
