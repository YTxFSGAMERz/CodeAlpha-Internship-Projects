# CodeAlpha Internship Projects

Hi, I'm Farhan Shaikh ([@YTxFSGAMERz](https://github.com/YTxFSGAMERz)). This repository contains the assignments and projects I completed during my internship with CodeAlpha.

The work is split into three tracks: **Artificial Intelligence**, **Cybersecurity**, and **Machine Learning**. Each folder is self-contained with its own code, data/assets, dependencies, and README.

---

## Projects Overview

### Artificial Intelligence (`/Artificial Intelligence`)

1. **Language Translation Tool** (`CodeAlpha_LanguageTranslationTool`)  
   A Flask web application that translates text across 100+ languages using `deep-translator` (Google Translate). Includes auto-language detection, browser-based text-to-speech, and copy/swap features.

2. **FAQ Chatbot** (`CodeAlpha_FAQChatbot`)  
   A customer support chatbot for an e-commerce store with both web and terminal interfaces. Uses NLTK for text preprocessing (tokenization, stop-word filtering, lemmatization), scikit-learn for TF-IDF vectorization, and cosine similarity to find matching answers or make suggestions.

3. **Music Generation with AI** (`CodeAlpha_MusicGenerationAI`)  
   An LSTM recurrent neural network built in Keras/PyTorch that learns musical patterns from MIDI files (Bach chorales). Given a starting seed phrase, it generates new polyphonic chord progressions and saves the result as both `.mid` and `.wav` audio.

4. **Object Detection and Tracking** (`CodeAlpha_ObjectDetectionTracking`)  
   Real-time video tracking that combines YOLOv8 object detection with a SORT tracker (Kalman Filter + Hungarian matching algorithm). Assigns persistent IDs to objects across frames even through temporary occlusions.

---

### Cybersecurity (`/Cybersecurity`)

1. **Basic Network Sniffer** (`CodeAlpha_BasicNetworkSniffer`)  
   A packet sniffer written in Python using Scapy. Captures and parses packets layer-by-layer (Ethernet, IPv4/IPv6, TCP/UDP, ICMP, DNS, HTTP, and TLS headers), with optional hex/ASCII payload previews and traffic breakdown statistics.

2. **Network Intrusion Detection System** (`CodeAlpha_NetworkIntrusionDetectionSystem`)  
   Custom Suricata 7 detection rules for port scanning, ping sweeps, SSH brute force, cleartext FTP/Telnet, and web attacks (SQL injection, directory traversal). Includes a Python script to synthesize realistic attack traffic into a replayable `.pcap` and an automated responder script.

3. **Phishing Awareness Training** (`CodeAlpha_PhishingAwarenessTraining`)  
   An interactive web application designed for employee security training. Users inspect simulated emails to find red flags (lookalike sender domains, mismatched links, urgent tone) and complete a scenario quiz with immediate explanations.

4. **Secure Code Review** (`CodeAlpha_SecureCodingReview`)  
   A security review of a sample Flask note-taking API ("NoteBox"). Covers 13 identified vulnerabilities (SQLi, command injection, insecure deserialization, IDOR, SSRF, CSRF, etc.), each backed by an automated exploit test, with hardened side-by-side fixes in `secure_app/`.

---

### Machine Learning (`/Machine Learning`)

1. **Credit Scoring Model** (`CodeAlpha_CreditScoringModel`)  
   Predicts loan default probability using 6 months of customer payment records and bills (UCI Default of Credit Card Clients dataset). Evaluates Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting, and maps default risk to a standard 300–850 credit scorecard.

2. **Disease Prediction from Medical Data** (`CodeAlpha_DiseasePrediction`)  
   Diagnostic machine learning pipelines across three UCI benchmark datasets: Heart Disease (Statlog), Diabetes (Pima Indians), and Breast Cancer (Wisconsin Diagnostic). Compares algorithms (SVM, Random Forest, XGBoost) and handles clinical feature preprocessing.

3. **Emotion Recognition from Speech** (`CodeAlpha_EmotionRecognitionFromSpeech`)  
   Speech emotion classifier that extracts 120 acoustic features (40 MFCCs plus delta and delta-delta derivatives) using Librosa. Includes audio data augmentation (noise, pitch shift, time stretch) and a synthetic demo dataset generator for smoke testing.

4. **Handwritten Character Recognition** (`CodeAlpha_HandwrittenCharacterRecognition`)  
   Computer vision OCR pipeline that takes raw images of handwriting, applies Otsu binarization and center-of-mass alignment into normalized $28 \times 28$ tensors, and segments multi-character images into individual letter bounding boxes.

---

## Setup & Running

### Requirements
- Python 3.10 to 3.13
- Git

### Quick Start

```bash
git clone https://github.com/YTxFSGAMERz/CodeAlpha-Internship-Projects.git
cd CodeAlpha-Internship-Projects

# Set up virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux / macOS:
source .venv/bin/activate

# Install dependencies for all tracks
pip install -r requirements.txt
```

Each individual project also has its own `requirements.txt` and a `README.md` with detailed options.

---

## Running the Tests

To run the automated tests across all 12 projects:

```bash
python run_all_tests.py
```

All 12 projects currently pass their test suites (121 tests total).

---

## Submission & Demos

For CodeAlpha task submissions, each project includes:
- Source code with explanatory comments
- README explaining how the approach works and how to run it
- Automated tests or sample scripts for live demonstration

See the project READMEs and the included PDF instruction sheets in each domain folder for details on recording demo walkthrough videos.

---

## Author

- **Farhan Shaikh**
- GitHub: [@YTxFSGAMERz](https://github.com/YTxFSGAMERz)
