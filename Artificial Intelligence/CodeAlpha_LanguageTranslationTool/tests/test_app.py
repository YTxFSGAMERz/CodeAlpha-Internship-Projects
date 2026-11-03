"""Tests for the translator backend. They never touch the network: the
translation service is replaced by a small fake."""

import sys
from pathlib import Path

import pytest
from deep_translator.exceptions import TooManyRequests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module  # noqa: E402
import translator  # noqa: E402


class FakeTranslator:
    calls = []

    def __init__(self, source="auto", target="en"):
        self.source, self.target = source, target

    def get_supported_languages(self, as_dict=False):
        return {"english": "en", "hindi": "hi", "french": "fr", "gujarati": "gu"}

    def translate(self, text):
        FakeTranslator.calls.append(text)
        return f"[{self.target}] {text}"


@pytest.fixture(autouse=True)
def fake_service(monkeypatch):
    FakeTranslator.calls = []
    translator._language_map.cache_clear()
    monkeypatch.setattr(translator, "GoogleTranslator", FakeTranslator)
    yield
    translator._language_map.cache_clear()


@pytest.fixture
def client():
    return app_module.app.test_client()


def post(client, **payload):
    return client.post("/api/translate", json=payload)


def test_home_page_lists_languages(client):
    html = client.get("/").get_data(as_text=True)
    assert "Detect language" in html
    assert 'value="hi" selected' in html
    assert "Gujarati" in html


def test_translates_text(client):
    response = post(client, text="Hello", source="auto", target="hi")
    assert response.status_code == 200
    assert response.get_json()["translation"] == "[hi] Hello"


def test_line_breaks_are_preserved(client):
    response = post(client, text="Good morning\n\nHow are you?", source="en", target="fr")
    assert response.get_json()["translation"] == "[fr] Good morning\n\n[fr] How are you?"


def test_repeated_lines_are_only_sent_once(client):
    post(client, text="Hi\nHi\nHi", source="en", target="fr")
    assert FakeTranslator.calls == ["Hi"]


def test_same_language_skips_the_service(client):
    response = post(client, text="Hello", source="en", target="en")
    assert response.get_json()["translation"] == "Hello"
    assert FakeTranslator.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "   ", "source": "en", "target": "fr"},
        {"text": "x" * (translator.MAX_CHARS + 1), "source": "en", "target": "fr"},
        {"text": "Hello", "source": "xx", "target": "fr"},
        {"text": "Hello", "source": "en", "target": "zz"},
        {"text": "\n".join(["line %d" % i for i in range(translator.MAX_LINES + 1)]), "source": "en", "target": "fr"},
    ],
)
def test_bad_input_is_rejected(client, payload):
    response = post(client, **payload)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_rate_limit_maps_to_429(client, monkeypatch):
    def boom(self, text):
        raise TooManyRequests()

    monkeypatch.setattr(FakeTranslator, "translate", boom)
    response = post(client, text="Hello", source="en", target="fr")
    assert response.status_code == 429
