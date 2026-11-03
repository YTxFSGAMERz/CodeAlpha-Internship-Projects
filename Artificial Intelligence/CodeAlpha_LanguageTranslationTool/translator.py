"""Translation logic for the Language Translation Tool.

The heavy lifting is done by the `deep-translator` package, which sends the
text to the Google Translate service and returns the translated result. It
needs no API key or billing account, so the project runs as soon as it is
installed. To switch to an official paid API (Google Cloud Translation or
Microsoft Translator) only `language_options()` and `_translate_line()` below
have to change - the web app and the UI stay exactly the same.
"""

from __future__ import annotations

from functools import lru_cache

import requests
from deep_translator import GoogleTranslator
from deep_translator.exceptions import (
    NotValidLength,
    NotValidPayload,
    RequestError,
    TooManyRequests,
    TranslationNotFound,
)

AUTO = "auto"          # let the service detect the source language
MAX_CHARS = 5000       # the service rejects anything longer than this
MAX_LINES = 40         # every non-empty line is translated with its own request


class TranslationError(Exception):
    """An error whose message is safe to show directly to the user."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@lru_cache(maxsize=1)
def _language_map() -> dict[str, str]:
    """Return {'hindi': 'hi', ...} for every language the service supports."""
    return GoogleTranslator(source=AUTO, target="en").get_supported_languages(as_dict=True)


def language_options() -> list[dict[str, str]]:
    """Languages as [{'code': 'hi', 'name': 'Hindi'}, ...], sorted by name."""
    options = [{"code": code, "name": name.title()} for name, code in _language_map().items()]
    return sorted(options, key=lambda option: option["name"])


def _translate_gtx(line: str, source: str, target: str) -> str:
    """Direct API fallback when web scraping encounters rate limits or errors."""
    url = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": source, "tl": target, "dt": "t", "q": line}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        if data and data[0]:
            return "".join(part[0] for part in data[0] if part and part[0])
    resp.raise_for_status()
    raise TranslationError("Could not translate this line.", 422)


def _translate_line(engine: GoogleTranslator, line: str, source: str = AUTO, target: str = "en") -> str:
    """Send one line of text to the translation service with an automatic fallback."""
    try:
        result = engine.translate(line.strip())
        if not result:
            raise TranslationError("The service returned an empty translation. Try rephrasing.", 422)
        return result
    except (TooManyRequests, RequestError):
        # When deep-translator gets 429 from Google's mobile web page, fall back to the API
        if type(engine).__name__ == "GoogleTranslator":
            try:
                return _translate_gtx(line.strip(), source, target)
            except Exception:
                pass
        raise


def translate(text: str, source: str = AUTO, target: str = "en") -> str:
    """Translate `text` from `source` to `target` and return the result.

    Line breaks are kept: each non-empty line is translated on its own and the
    lines are joined back together in the original order.
    """
    text = (text or "").strip()
    if not text:
        raise TranslationError("Enter some text to translate.")
    if len(text) > MAX_CHARS:
        raise TranslationError(f"That text is too long. The limit is {MAX_CHARS:,} characters.")

    codes = set(_language_map().values())
    if source != AUTO and source not in codes:
        raise TranslationError("The source language is not supported.")
    if target not in codes:
        raise TranslationError("The target language is not supported.")
    if source == target:
        return text

    lines = text.splitlines()
    if sum(1 for line in lines if line.strip()) > MAX_LINES:
        raise TranslationError(f"Too many lines. Keep it under {MAX_LINES} lines or translate in parts.")

    engine = GoogleTranslator(source=source, target=target)
    done: dict[str, str] = {}   # repeated lines are only sent once
    translated: list[str] = []
    try:
        for line in lines:
            if not line.strip():
                translated.append("")
                continue
            if line not in done:
                done[line] = _translate_line(engine, line, source, target)
            translated.append(done[line])
    except TranslationError:
        raise
    except TooManyRequests as err:
        raise TranslationError("Too many requests. Wait a few seconds and try again.", 429) from err
    except (RequestError, requests.RequestException) as err:
        raise TranslationError(
            "Could not reach the translation service. Check your internet connection.", 502
        ) from err
    except (TranslationNotFound, NotValidPayload, NotValidLength) as err:
        raise TranslationError("The service could not translate this text.", 422) from err
    except Exception as err:  # keep the web app alive whatever the library throws
        raise TranslationError("Something went wrong while translating. Try again.", 500) from err

    return "\n".join(translated)
