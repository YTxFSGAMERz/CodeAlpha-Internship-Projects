# CodeAlpha ML internship - submission checklist

All four tasks are built as separate projects. CodeAlpha asks for 2-3 tasks, so you can submit all four or pick the ones
you like best. Each folder is named exactly like the GitHub repository CodeAlpha wants (`CodeAlpha_ProjectName`).

| Folder = repository name | Task |
|---|---|
| `CodeAlpha_CreditScoringModel` | Task 1 - Credit Scoring Model |
| `CodeAlpha_EmotionRecognitionFromSpeech` | Task 2 - Emotion Recognition from Speech |
| `CodeAlpha_HandwrittenCharacterRecognition` | Task 3 - Handwritten Character Recognition |
| `CodeAlpha_DiseasePrediction` | Task 4 - Disease Prediction from Medical Data |

## 1. Run every project yourself first (you will explain it on video)

Python 3.10-3.12 is the safest choice for TensorFlow. For each folder: `pip install -r requirements.txt`, then

```bash
# Task 1 and Task 4 - quick, no downloads
python train.py --quick          # or without --quick for the full run
python predict.py ...            # examples are in each README

# Task 3 - downloads MNIST automatically (EMNIST: pip install emnist, ~535 MB once)
python train.py --dataset mnist

# Task 2 - download at least one dataset first (search for the name):
#   RAVDESS: "Audio_Speech_Actors_01-24.zip" (Zenodo)   TESS: Toronto Emotional Speech Set (Kaggle)
#   EMO-DB: Berlin Database of Emotional Speech
python tools/make_demo_data.py && python train.py --ravdess demo_data/RAVDESS --epochs 15   # 1-minute smoke test
python train.py --ravdess path/to/RAVDESS --augment 2
```

Your numbers will differ a little from the ones in the READMEs - that is normal. Put your own final numbers in the
README "Results" section (the speech README has a table waiting for them).

## 2. Put each project on GitHub (one repository per folder)

Create an empty repository with the exact folder name on github.com, then:

```bash
cd CodeAlpha_CreditScoringModel
git init
git add .
git commit -m "CodeAlpha ML internship - Credit Scoring Model"
git branch -M main
git remote add origin https://github.com/<your-username>/CodeAlpha_CreditScoringModel.git
git push -u origin main
```

(`.gitignore` already keeps trained models and caches out of the repository.)

## 3. LinkedIn

1. Post your internship status and tag **@CodeAlpha**.
2. Record a 2-4 minute video per project and post it with the GitHub link. A simple outline:
   problem -> dataset -> approach (show the pipeline in the README) -> run the code live -> read the results
   (confusion matrix / ROC curve) -> one limitation and one improvement you would make.

## 4. Submit

Use the submission form shared in your CodeAlpha WhatsApp group - only that form is accepted.
