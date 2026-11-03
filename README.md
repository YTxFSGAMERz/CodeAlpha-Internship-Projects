# CodeAlpha Internship Projects - Master Portfolio

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black.svg)](https://flask.palletsprojects.com/)
[![Scapy](https://img.shields.io/badge/Scapy-2.5%2B-red.svg)](https://scapy.net/)
[![Tests](https://img.shields.io/badge/Tests-121%20Passed%20(100%25)-brightgreen.svg)](#master-test-runner)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A unified repository containing coursework and project implementations for the **CodeAlpha Internship** completed by **[Farhan (YTxFSGAMERz)](https://github.com/YTxFSGAMERz)**. 

This repository spans **3 distinct engineering tracks** and **12 production-ready projects**, complete with interactive web interfaces, machine learning models, neural networks, network packet analyzers, intrusion detection signatures, and automated test suites.

---

## 🏛️ Repository Architecture

```
CodeAlpha Internship Projects/
├── verify_all.py                       # Master verification test runner
├── .gitignore                          # Global monorepo ignore rules
│
├── Artificial Intelligence/            # Track 1: AI & NLP Systems
│   ├── CodeAlpha_FAQChatbot/           # Task 1: NLP FAQ Chatbot + Web App
│   ├── CodeAlpha_LanguageTranslationTool/ # Task 2: Multi-Language Translator + Web App
│   ├── CodeAlpha_MusicGenerationAI/    # Task 3: Deep LSTM MIDI & Audio Composer
│   └── CodeAlpha_ObjectDetectionTracking/ # Task 4: YOLOv8 + SORT Multi-Object Tracker
│
├── Cybersecurity/                      # Track 2: Security & Network Defense
│   ├── CodeAlpha_BasicNetworkSniffer/  # Task 1: Raw Packet Sniffer & Protocol Decoder
│   ├── CodeAlpha_NetworkIntrusionDetectionSystem/ # Task 2: Signature NIDS & Traffic Analyzer
│   ├── CodeAlpha_PhishingAwarenessTraining/ # Task 3: Interactive Web Phishing Simulator
│   └── CodeAlpha_SecureCodingReview/   # Task 4: NoteBox Vulnerable vs Hardened API (OWASP)
│
└── Machine Learning/                   # Track 3: Predictive ML & Computer Vision
    ├── CodeAlpha_CreditScoringModel/   # Task 1: Credit Risk Scorer & Financial Scorecard
    ├── CodeAlpha_DiseasePrediction/    # Task 2: Multi-Disease Clinical Diagnostic Models
    ├── CodeAlpha_EmotionRecognitionFromSpeech/ # Task 3: Audio MFCC + CNN/LSTM Emotion Classifier
    └── CodeAlpha_HandwrittenCharacterRecognition/ # Task 4: CNN OCR & Handwriting Segmentation
```

---

## 🚀 Projects Overview & Showcase

### 1. Artificial Intelligence Track

| Task | Project | Stack | Description | Status |
|:---:|:---|:---|:---|:---:|
| **01** | [`CodeAlpha_LanguageTranslationTool`](./Artificial%20Intelligence/CodeAlpha_LanguageTranslationTool) | Flask, Deep-Translator, JS | Real-time multi-language translation platform with modern responsive web UI. | Verified (11 tests) |
| **02** | [`CodeAlpha_FAQChatbot`](./Artificial%20Intelligence/CodeAlpha_FAQChatbot) | Flask, NLTK, Scikit-Learn | Intent-aware customer support chatbot using TF-IDF and cosine similarity. | Verified (31 tests) |
| **03** | [`CodeAlpha_MusicGenerationAI`](./Artificial%20Intelligence/CodeAlpha_MusicGenerationAI) | Keras 3, PyTorch/TF, music21 | Deep LSTM neural network composing novel polyphonic piano pieces into MIDI and synthesized WAV audio. | Verified (13 tests) |
| **04** | [`CodeAlpha_ObjectDetectionTracking`](./Artificial%20Intelligence/CodeAlpha_ObjectDetectionTracking) | YOLOv8, OpenCV, SORT | Real-time video multi-object detection and tracking using Kalman filtering and Hungarian association. | Verified (19 tests) |

---

### 2. Cybersecurity Track

| Task | Project | Stack | Description | Status |
|:---:|:---|:---|:---|:---:|
| **01** | [`CodeAlpha_BasicNetworkSniffer`](./Cybersecurity/CodeAlpha_BasicNetworkSniffer) | Python 3, Scapy | Live and offline network packet sniffer decoding Ethernet, IPv4/v6, TCP/UDP, ICMP, DNS, and HTTP/TLS. | Verified (6 tests) |
| **02** | [`CodeAlpha_NetworkIntrusionDetectionSystem`](./Cybersecurity/CodeAlpha_NetworkIntrusionDetectionSystem) | Scapy, Suricata, Matplotlib | Signature-based network intrusion detection system with synthetic attack generation and incident response. | Verified (2 tests) |
| **03** | [`CodeAlpha_PhishingAwarenessTraining`](./Cybersecurity/CodeAlpha_PhishingAwarenessTraining) | HTML5, Modern CSS, Vanilla JS | Interactive training simulator teaching employees to detect email red flags, spear phishing, and spoofed links. | Verified (2 tests) |
| **04** | [`CodeAlpha_SecureCodingReview`](./Cybersecurity/CodeAlpha_SecureCodingReview) | Python, Flask, Bandit, Pytest | Complete security audit of NoteBox API uncovering 13 vulnerabilities (SQLi, RCE, Deserialization, SSRF, IDOR) with side-by-side hardened fixes. | Verified (26 tests) |

---

### 3. Machine Learning Track

| Task | Project | Stack | Description | Status |
|:---:|:---|:---|:---|:---:|
| **01** | [`CodeAlpha_CreditScoringModel`](./Machine%20Learning/CodeAlpha_CreditScoringModel) | Scikit-Learn, Pandas, Joblib | Credit default prediction pipeline comparing Logistic Regression, Random Forest, and Gradient Boosting (ROC-AUC: 0.786) with credit scorecards. | Verified (3 tests) |
| **02** | [`CodeAlpha_DiseasePrediction`](./Machine%20Learning/CodeAlpha_DiseasePrediction) | Scikit-Learn, XGBoost, Pandas | Predictive diagnostic models trained across 3 distinct UCI clinical datasets (Heart Disease, Diabetes, Breast Cancer) achieving up to 0.995 ROC-AUC. | Verified (2 tests) |
| **03** | [`CodeAlpha_EmotionRecognitionFromSpeech`](./Machine%20Learning/CodeAlpha_EmotionRecognitionFromSpeech) | Librosa, SoundFile, NumPy | Speech emotion classification extracting 120-dimensional perceptual acoustic features (MFCCs, deltas, delta-deltas) with audio augmentation. | Verified (2 tests) |
| **04** | [`CodeAlpha_HandwrittenCharacterRecognition`](./Machine%20Learning/CodeAlpha_HandwrittenCharacterRecognition) | OpenCV, NumPy, SciPy, Pillow | Computer vision OCR pipeline performing Otsu binarization, bounding-box segmentation, and MNIST/EMNIST character recognition. | Verified (4 tests) |

---

## 🧪 Master Test Runner

This monorepo includes a central verification engine [`verify_all.py`](./verify_all.py) that automatically validates all 12 projects:

```bash
python verify_all.py
```

### Verification Output:
```text
================================================================================
  CODEALPHA INTERNSHIP PROJECTS - MASTER VERIFICATION RUNNER
================================================================================
Repository Root: C:\Users\Admin\Documents\GitHub\CodeAlpha Internship Projects

Track                     Project                                  Status     Tests    Time
------------------------------------------------------------------------------------------
Artificial Intelligence   CodeAlpha_LanguageTranslationTool        PASSED     11       0.94s
Artificial Intelligence   CodeAlpha_FAQChatbot                     PASSED     31       4.72s
Artificial Intelligence   CodeAlpha_MusicGenerationAI              PASSED     13       7.19s
Artificial Intelligence   CodeAlpha_ObjectDetectionTracking        PASSED     19       1.35s
Cybersecurity             CodeAlpha_BasicNetworkSniffer            PASSED     6        1.43s
Cybersecurity             CodeAlpha_NetworkIntrusionDetectionSystem PASSED     2        1.44s
Cybersecurity             CodeAlpha_PhishingAwarenessTraining      PASSED     2        0.67s
Cybersecurity             CodeAlpha_SecureCodingReview             PASSED     26       6.69s
Machine Learning          CodeAlpha_CreditScoringModel             PASSED     3        2.29s
Machine Learning          CodeAlpha_DiseasePrediction              PASSED     2        2.20s
Machine Learning          CodeAlpha_EmotionRecognitionFromSpeech   PASSED     2        2.20s
Machine Learning          CodeAlpha_HandwrittenCharacterRecognition PASSED     4        0.95s
------------------------------------------------------------------------------------------
Total Projects Verified: 12/12
Total Automated Tests Passed: 121
Total Failures: 0
Total Duration: 32.08s
================================================================================

All 12 CodeAlpha projects are healthy and fully verified!
```

---

## 💻 Quickstart & Setup

### Prerequisites
- Python 3.10 – 3.13
- Git

### Installation
Clone the repository:
```bash
git clone https://github.com/YTxFSGAMERz/CodeAlpha-Internship-Projects.git
cd CodeAlpha-Internship-Projects
```

Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate
```

Install common development dependencies:
```bash
pip install -r requirements.txt
```
*(Or navigate into any project folder to install its specific `requirements.txt`).*

---

## 📹 Video Demonstration & Submission Guide

For each task submitted to CodeAlpha:
1. **Walkthrough Video**: Record a 2–4 minute video demonstrating:
   - Problem statement & dataset / protocol explanation
   - Live code execution and UI demo
   - Model metrics / attack reproduction results
   - Limitations and production improvements
2. **LinkedIn Post**: Share your achievements tagging `@CodeAlpha` with your demo video and GitHub link.
3. **Submission Form**: Submit via the official CodeAlpha portal / WhatsApp group link.

Detailed demonstration commands for every single project are documented in the [Walkthrough Guide](./walkthrough.md).

---

## 👤 Author
- **Farhan** ([@YTxFSGAMERz](https://github.com/YTxFSGAMERz))
- CodeAlpha Internship Program
