# FAQ Chatbot

**CodeAlpha Artificial Intelligence Internship - Task 2**

A chatbot that answers frequently asked questions. You type a question in your own words, and the bot finds the most similar FAQ and replies with its answer. The sample data is the help centre of a made-up online store called **ShopEasy** (orders, delivery, returns, payments, account), and you can swap in your own FAQs by editing one JSON file.

It comes with two ways to chat: a web chat window and a terminal version.

## How it works

```
user question
   │
   ├─ greeting / thanks / goodbye / "who are you"?  ──► intent rules (regex)  ──► canned reply
   │
   ▼
1. Preprocess (NLTK)   lower-case → expand "can't" → tokenize → remove stop words → lemmatize
2. Vectorize           TF-IDF vector of the cleaned question (scikit-learn)
3. Match               cosine similarity against every FAQ question and its alternative phrasings
4. Decide              confident?  → answer
                       unsure?     → "Did you mean...?" with the closest questions
                       no idea?    → polite fallback with popular questions
```

- **Preprocessing** (`FAQChatbot.preprocess`): `word_tokenize` splits the text, NLTK's English stop words are removed (but words like *not*, *where*, *when*, *how* are kept because they change the meaning), and `WordNetLemmatizer` reduces words to their base form, so "orders", "ordered" and "ordering" all become `order`.
- **TF-IDF** gives rare, meaningful words (such as *refund*) more weight than common ones (such as *order*).
- **Cosine similarity** measures how close the question is to each FAQ. Every FAQ has several rows (its question, alternative phrasings and keywords) and keeps its best score.
- **Confidence**: the bot only answers when the similarity is at least `0.30` and clearly ahead of the runner-up. The score is also lowered when many words in the question are unknown to the FAQ data, so unrelated questions are less likely to get a confident wrong answer. These numbers are constants at the top of `chatbot.py`.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python app.py                 # web chat at http://127.0.0.1:5000
```

The first start downloads three small NLTK data packages (`punkt_tab`, `stopwords`, `wordnet`), so an internet connection is needed once.

Terminal version:

```bash
python chatbot.py                                  # interactive chat
python chatbot.py "how long does delivery take"    # one question, shows the similarity score
```

## Use your own FAQs

Edit `data/faqs.json`. Each FAQ looks like this:

```json
{
  "id": "refund-time",
  "category": "Returns",
  "suggested": true,
  "question": "When will I get my refund?",
  "variants": ["How long does a refund take?", "Where is my refund?"],
  "keywords": "refund money back credited status",
  "answer": "Refunds are issued 2-3 business days after ..."
}
```

- `variants` are other ways people ask the same thing. More variants mean better matching - this is the single best way to improve the bot.
- `keywords` is an optional list of related words.
- `suggested: true` shows the question as a starter button in the chat window.
- Small talk (hello, thanks, bye, "who are you") is defined under `intents` with regular-expression patterns that must match the whole message.

## Project structure

```
CodeAlpha_FAQChatbot/
├── chatbot.py          # the engine: preprocessing, TF-IDF, cosine similarity, terminal chat
├── app.py              # Flask server for the web chat
├── data/faqs.json      # FAQs and small-talk intents
├── templates/index.html
├── static/style.css
├── static/app.js       # chat window behaviour
├── tests/test_chatbot.py
└── requirements.txt
```

## Tests

```bash
pytest
```

## How well does it work?

Honest numbers from my own checks: after tuning, I wrote 30 new questions in everyday wording that the bot had never seen. It answered 17 correctly and 3 wrongly, and for the other 10 it said it was unsure or did not know (in 4 of those the right FAQ was among the suggested questions). Out of 10 unrelated questions, 4 still got an answer because they shared a word with an FAQ (for example "how to apply for a passport" matched the coupon FAQ, because of the word *apply*).

That is typical for a word-matching approach: it cannot understand meaning, so questions that use different words from the FAQs ("shirt reaches me" instead of "order arrives") are the hardest. Ways to improve it:

- add more `variants` for the FAQs that get missed,
- replace TF-IDF with sentence embeddings (for example `sentence-transformers`) while keeping the same cosine-similarity step,
- lower or raise `CONFIDENCE_THRESHOLD` and `MARGIN` in `chatbot.py` to trade fewer wrong answers against more "did you mean" replies.
