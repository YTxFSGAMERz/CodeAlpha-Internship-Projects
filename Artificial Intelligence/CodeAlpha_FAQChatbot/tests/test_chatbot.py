"""Tests for the FAQ chatbot engine and its web API. Run with: pytest"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import FAQChatbot  # noqa: E402
import app as app_module  # noqa: E402


@pytest.fixture(scope="module")
def bot():
    return FAQChatbot()


# ---------------------------------------------------------------- preprocessing
def test_preprocess_cleans_and_lemmatizes(bot):
    assert bot.preprocess("My ORDERS were delayed!!!") == "order delay"


def test_contractions_keep_the_negation(bot):
    assert "not" in bot.preprocess("I can't log in").split()


# --------------------------------------------------------------------- matching
PARAPHRASES = [
    ("how long will it take to get my order", "How long does delivery take?"),
    ("what are the shipping fees", "How much does shipping cost?"),
    ("track order", "How can I track my order?"),
    ("what is the return policy", "What is your return policy?"),
    ("i want my money back", "When will I get my refund?"),
    ("do you accept upi", "Which payment methods do you accept?"),
    ("is cod possible", "Is cash on delivery available?"),
    ("forgot password", "I forgot my password. How do I reset it?"),
    ("i need a gst bill", "How can I download my invoice or GST bill?"),
    ("do you ship outside india", "Do you ship internationally?"),
]


@pytest.mark.parametrize("question, expected", PARAPHRASES)
def test_paraphrases_find_the_right_faq(bot, question, expected):
    reply = bot.ask(question)
    assert reply["source"] == "faq"
    assert reply["matched_question"] == expected
    assert reply["score"] >= bot.threshold


def test_exact_question_scores_highest(bot):
    reply = bot.ask("What is your return policy?")
    assert reply["score"] == pytest.approx(1.0, abs=0.01)


def test_greeting_prefix_is_ignored(bot):
    reply = bot.ask("Hi, how long does delivery take?")
    assert reply["matched_question"] == "How long does delivery take?"


# ---------------------------------------------------------------------- intents
@pytest.mark.parametrize("message", ["hello", "Good morning!", "thanks a lot", "bye", "who are you?", "what can you do"])
def test_small_talk_is_handled_by_intents(bot, message):
    reply = bot.ask(message)
    assert reply["source"] == "intent"
    assert reply["confident"]


# ------------------------------------------------------------- not sure / unknown
@pytest.mark.parametrize("message", ["what is the capital of france", "asdf qwerty", "recommend a good movie", "", "   "])
def test_unrelated_questions_fall_back(bot, message):
    reply = bot.ask(message)
    assert reply["source"] == "fallback"
    assert not reply["confident"]
    assert reply["answer"] == bot.fallback
    assert reply["suggestions"] == bot.suggested_questions()


def test_unsure_match_offers_suggestions_instead_of_guessing(bot):
    reply = bot.ask("will this laptop be covered by warranty")
    assert reply["source"] == "suggestions"
    assert not reply["confident"]
    assert reply["suggestions"][0] == "Do products come with a warranty?"


# --------------------------------------------------------------------- web API
@pytest.fixture
def client():
    return app_module.app.test_client()


def test_home_page_contains_starter_questions(client):
    html = client.get("/").get_data(as_text=True)
    assert "ShopEasy Help" in html
    assert "How long does delivery take?" in html


def test_chat_endpoint_returns_a_reply(client):
    response = client.post("/api/chat", json={"message": "how can i track my package"})
    data = response.get_json()
    assert response.status_code == 200
    assert data["source"] == "faq"
    assert "My Orders" in data["answer"]


@pytest.mark.parametrize("payload", [{}, {"message": "   "}, {"message": "x" * 500}])
def test_chat_endpoint_rejects_bad_input(client, payload):
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"]
