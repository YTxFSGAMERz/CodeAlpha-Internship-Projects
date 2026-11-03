"""FAQ chatbot engine.

How a question is answered
--------------------------
1. Intent rules - greetings, thanks, goodbyes and "who are you" are recognised
                  with simple patterns (intent matching).
2. Preprocess   - lower-case, expand contractions, tokenize, drop stop words and
                  reduce each word to its base form (NLTK).
3. Vectorize    - turn the text into a TF-IDF vector (scikit-learn).
4. Match        - cosine similarity between the user's vector and every FAQ
                  question (including its alternative phrasings).
5. Answer       - the best FAQ is returned if the bot is confident. If two FAQs
                  are about equally likely, or the words are mostly unknown, the
                  bot admits it is unsure and offers the closest questions.

Run it in the terminal:   python chatbot.py
Ask a single question:    python chatbot.py "how long does delivery take"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path(__file__).resolve().parent / "data" / "faqs.json"

CONFIDENCE_THRESHOLD = 0.30   # below this similarity the bot does not answer
STRONG_MATCH = 0.80           # at or above this the bot answers even if another FAQ is close
MARGIN = 0.04                 # otherwise the best FAQ must beat the runner-up by this much
SUGGESTION_THRESHOLD = 0.20   # below this a question is not worth suggesting
MAX_SUGGESTIONS = 3
NGRAM_RANGE = (1, 1)          # single words; (1, 2) would add word pairs but generalised worse

UNSURE_MESSAGE = "I'm not completely sure what you mean. Did you mean one of these?"

# NLTK package -> folder that proves it is installed
NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}

# Stop words that carry meaning in a support chat ("can not log in", "out of stock",
# "where is my order" vs "when will my order arrive")
KEEP_WORDS = {
    "no", "not", "nor", "out", "off", "after", "before",
    "where", "when", "how", "why", "what", "which", "who",
}

# Filler words that appear in many questions and say nothing about the topic
EXTRA_STOP_WORDS = {
    "please", "tell", "want", "need", "would", "could", "like", "get", "got",
    "know", "let", "also", "just", "something", "thing",
}

_CONTRACTIONS = [
    (r"\bcan't\b", "can not"),
    (r"\bwon't\b", "will not"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'ll\b", " will"),
    (r"'ve\b", " have"),
    (r"'m\b", " am"),
    (r"'d\b", " would"),
    (r"'s\b", ""),
]

# "Hi, how long does delivery take?" -> "how long does delivery take?"
_GREETING_PREFIX = re.compile(
    r"^\s*(?:hi+|hello+|hey+|hola|namaste|good\s+(?:morning|afternoon|evening))\b"
    r"[\s,!.:-]*(?:(?:there|team|bot|shopeasy)\b[\s,!.:-]*)?"
)


def ensure_nltk_data() -> None:
    """Download the NLTK data this project needs (only the first time)."""
    missing = []
    for package, path in NLTK_RESOURCES.items():
        for candidate in (path, path + ".zip"):
            try:
                nltk.data.find(candidate)
                break
            except LookupError:
                continue
        else:
            if not nltk.download(package, quiet=True):
                missing.append(package)
    if missing:
        raise SystemExit(
            "Could not download the NLTK data (" + ", ".join(missing) + "). Connect to the internet and try again, "
            "or run:  python -m nltk.downloader " + " ".join(missing)
        )


def _plain(text: str) -> str:
    """Lower-case, drop apostrophes and punctuation: "That's all!" -> "thats all"."""
    text = text.lower().replace("\u2019", "'").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text).split())


class FAQChatbot:
    def __init__(
        self,
        data_file: Path | str = DATA_FILE,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        ensure_nltk_data()
        self.threshold = threshold
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = (set(stopwords.words("english")) - KEEP_WORDS) | EXTRA_STOP_WORDS

        with open(data_file, encoding="utf-8") as handle:
            data = json.load(handle)
        self.store = data.get("store", "Support")
        self.fallback = data.get("fallback", "Sorry, I couldn't find an answer to that.")
        self.faqs = data["faqs"]
        self._intents = [
            (intent, [re.compile(p) for p in intent["patterns"]])
            for intent in data.get("intents", [])
        ]

        # Every FAQ contributes several "rows": its question, alternative
        # phrasings and a keyword string. A user question is compared to all rows
        # and each FAQ keeps the best score of its rows.
        rows, row_faq = [], []
        for index, faq in enumerate(self.faqs):
            texts = [faq["question"], *faq.get("variants", [])]
            if faq.get("keywords"):
                texts.append(faq["keywords"])
            for text in texts:
                rows.append(self._clean(text))
                row_faq.append(index)
        self._row_faq = np.array(row_faq)

        self._vectorizer = TfidfVectorizer(
            ngram_range=NGRAM_RANGE, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"
        )
        self._matrix = self._vectorizer.fit_transform(rows)
        self._vocabulary = set(self._vectorizer.vocabulary_)

    # ------------------------------------------------------------------ NLP
    def preprocess(self, text: str, remove_stop_words: bool = True) -> str:
        """Clean `text` and return its tokens joined by spaces."""
        text = text.lower().replace("\u2019", "'")
        for pattern, replacement in _CONTRACTIONS:
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        tokens = word_tokenize(text)
        if remove_stop_words:
            tokens = [t for t in tokens if t not in self._stop_words]
        lemmas = []
        for token in tokens:
            lemma = self._lemmatizer.lemmatize(token, pos="v")
            lemmas.append(self._lemmatizer.lemmatize(lemma, pos="n"))
        return " ".join(lemmas)

    def _clean(self, text: str) -> str:
        """Preprocess, but keep stop words if nothing else would be left ("what is your name")."""
        return self.preprocess(text) or self.preprocess(text, remove_stop_words=False)

    # --------------------------------------------------------------- answers
    def _match_intent(self, text: str) -> dict | None:
        plain = _plain(text)
        for intent, patterns in self._intents:
            if plain and any(p.fullmatch(plain) for p in patterns):
                return intent
        return None

    def _scores(self, message: str) -> np.ndarray:
        """Confidence for each FAQ: cosine similarity, lowered when words are unknown."""
        processed = self._clean(message)
        tokens = processed.split()
        if not tokens:
            return np.zeros(len(self.faqs))
        known = sum(token in self._vocabulary for token in tokens) / len(tokens)

        vector = self._vectorizer.transform([processed])
        row_scores = cosine_similarity(vector, self._matrix).ravel()
        scores = np.zeros(len(self.faqs))
        np.maximum.at(scores, self._row_faq, row_scores)
        return scores * known

    def ask(self, message: str) -> dict:
        """Return the bot's reply to `message` as a dictionary."""
        message = (message or "").strip()

        intent = self._match_intent(message)
        if intent is None:
            remainder = _GREETING_PREFIX.sub("", message.lower()).strip()
            if remainder != message.lower():
                intent = self._match_intent(remainder)
                message = remainder or message
        if intent is not None:
            return self._reply(intent["answer"], source="intent", score=1.0, category="Small talk")

        scores = self._scores(message) if message else np.zeros(len(self.faqs))
        order = np.argsort(-scores)
        best = float(scores[order[0]])
        runner_up = float(scores[order[1]]) if len(order) > 1 else 0.0

        if best >= self.threshold and (best >= STRONG_MATCH or best - runner_up >= MARGIN):
            faq = self.faqs[int(order[0])]
            return self._reply(
                faq["answer"], source="faq", score=best,
                matched_question=faq["question"], category=faq.get("category"),
            )

        suggestions = [
            self.faqs[int(i)]["question"]
            for i in order[:MAX_SUGGESTIONS]
            if scores[int(i)] >= SUGGESTION_THRESHOLD
        ]
        if suggestions:
            return self._reply(UNSURE_MESSAGE, source="suggestions", score=best, suggestions=suggestions)
        return self._reply(
            self.fallback, source="fallback", score=best, suggestions=self.suggested_questions()
        )

    @staticmethod
    def _reply(answer, source, score, matched_question=None, category=None, suggestions=None) -> dict:
        return {
            "answer": answer,
            "source": source,                 # "faq", "intent", "suggestions" or "fallback"
            "confident": source in {"faq", "intent"},
            "score": round(float(score), 3),
            "matched_question": matched_question,
            "category": category,
            "suggestions": suggestions or [],   # questions the user can tap next
        }

    def suggested_questions(self) -> list[str]:
        """Starter questions shown on the chat page."""
        return [faq["question"] for faq in self.faqs if faq.get("suggested")]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    bot = FAQChatbot()

    if len(sys.argv) > 1:  # one-shot mode, handy for testing
        reply = bot.ask(" ".join(sys.argv[1:]))
        print(f"{reply['answer']}\n(source: {reply['source']}, similarity {reply['score']:.2f}, "
              f"matched: {reply['matched_question']})")
        return

    print(f"{bot.store} help bot. Type your question, or 'quit' to leave.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"quit", "exit", "q"}:
            break
        if not message:
            continue
        reply = bot.ask(message)
        print(f"Bot: {reply['answer']}")
        if reply["source"] == "faq":
            print(f"     [matched: {reply['matched_question']} | similarity {reply['score']:.2f}]")
        if reply["source"] == "fallback":
            print("     You could try:")
        for question in reply["suggestions"]:
            print(f"       - {question}")
        print()


if __name__ == "__main__":
    main()
