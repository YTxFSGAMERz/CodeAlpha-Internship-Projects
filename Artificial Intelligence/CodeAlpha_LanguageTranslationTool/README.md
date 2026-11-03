# Language Translation Tool

**CodeAlpha Artificial Intelligence Internship - Task 1**

A small web app where you type text, pick a source and a target language, and read the translation next to it. The text is sent to the Google Translate service through the [`deep-translator`](https://pypi.org/project/deep-translator/) package, so it works without an API key or billing account.

## Features

- Text box with a character counter (5,000 character limit, which is the service limit)
- Source language (with **Detect language**) and target language, 100+ languages
- **Translate** button, or press `Ctrl + Enter` (`Cmd + Enter` on Mac)
- **Copy** button for the translated text
- **Listen** button that reads the translation aloud (text-to-speech built into the browser)
- **Swap** button that flips the languages and moves the translation back into the input
- Changing the target language after a translation re-translates automatically
- Line breaks are kept, errors are explained in plain words, works in light and dark mode

## Tech stack

| Part | Tool |
| --- | --- |
| Backend | Python 3, Flask |
| Translation | `deep-translator` (Google Translate) |
| Frontend | HTML, CSS, vanilla JavaScript |
| Text-to-speech | Browser Web Speech API |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. An internet connection is needed because the text is translated online.

## How it works

1. The page (`templates/index.html`) is filled with the language list from `translator.language_options()`.
2. When you press **Translate**, `static/app.js` sends `{ text, source, target }` to `POST /api/translate`.
3. `app.py` passes the request to `translator.translate()`, which validates the input, translates every non-empty line with `GoogleTranslator`, and joins the lines back together.
4. The JSON reply `{ "translation": "..." }` (or `{ "error": "..." }`) is shown in the right-hand page.

You can also call the API directly:

```bash
curl -X POST http://127.0.0.1:5000/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Good morning", "source": "en", "target": "hi"}'
```

## Project structure

```
CodeAlpha_LanguageTranslationTool/
├── app.py              # Flask routes
├── translator.py       # translation logic, validation, error handling
├── templates/index.html
├── static/style.css
├── static/app.js       # UI behaviour: translate, copy, listen, swap
├── tests/test_app.py   # backend tests (no internet needed)
└── requirements.txt
```

## Tests

```bash
pytest
```

The tests replace the translation service with a fake, so they run offline.

## Notes and limitations

- `deep-translator` uses the public Google Translate web endpoint. That is fine for a project like this, but it is not an official API and Google can rate-limit it (the app then shows "Too many requests"). For production use an official API with a key, such as Google Cloud Translation or Microsoft Translator. Only `language_options()` and `_translate_line()` in `translator.py` would need to change.
- **Listen** uses the voices installed in your browser and operating system. If there is no voice for a language, the app says so.
- **Swap** is disabled while the source language is *Detect language*, because the app does not know which language was detected.
