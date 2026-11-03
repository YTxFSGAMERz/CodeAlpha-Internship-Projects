#!/usr/bin/env python3
"""
Test runner for all CodeAlpha internship projects.
Runs the pytest suite in each project folder and reports the results.
"""
import sys
import time
import subprocess
from pathlib import Path

# Fix terminal encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent

PROJECTS = [
    # AI Tasks
    ("Artificial Intelligence", "CodeAlpha_LanguageTranslationTool", "Artificial Intelligence/CodeAlpha_LanguageTranslationTool/tests"),
    ("Artificial Intelligence", "CodeAlpha_FAQChatbot", "Artificial Intelligence/CodeAlpha_FAQChatbot/tests"),
    ("Artificial Intelligence", "CodeAlpha_MusicGenerationAI", "Artificial Intelligence/CodeAlpha_MusicGenerationAI/tests"),
    ("Artificial Intelligence", "CodeAlpha_ObjectDetectionTracking", "Artificial Intelligence/CodeAlpha_ObjectDetectionTracking/tests"),

    # Cybersecurity Tasks
    ("Cybersecurity", "CodeAlpha_BasicNetworkSniffer", "Cybersecurity/CodeAlpha_BasicNetworkSniffer/tests"),
    ("Cybersecurity", "CodeAlpha_NetworkIntrusionDetectionSystem", "Cybersecurity/CodeAlpha_NetworkIntrusionDetectionSystem/tests"),
    ("Cybersecurity", "CodeAlpha_PhishingAwarenessTraining", "Cybersecurity/CodeAlpha_PhishingAwarenessTraining/tests"),
    ("Cybersecurity", "CodeAlpha_SecureCodingReview", "Cybersecurity/CodeAlpha_SecureCodingReview/tests"),

    # Machine Learning Tasks
    ("Machine Learning", "CodeAlpha_CreditScoringModel", "Machine Learning/CodeAlpha_CreditScoringModel/tests"),
    ("Machine Learning", "CodeAlpha_DiseasePrediction", "Machine Learning/CodeAlpha_DiseasePrediction/tests"),
    ("Machine Learning", "CodeAlpha_EmotionRecognitionFromSpeech", "Machine Learning/CodeAlpha_EmotionRecognitionFromSpeech/tests"),
    ("Machine Learning", "CodeAlpha_HandwrittenCharacterRecognition", "Machine Learning/CodeAlpha_HandwrittenCharacterRecognition/tests"),
]


def main():
    print("Running tests across all CodeAlpha projects...\n")
    start_time = time.time()
    passed_total = 0
    failed_projects = []

    for track, name, test_dir in PROJECTS:
        test_path = ROOT / test_dir
        if not test_path.exists():
            print(f"[-] {name:<42} (no test folder found)")
            continue

        t0 = time.time()
        res = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-q"],
            cwd=str(ROOT),
            capture_output=True,
            text=True
        )
        elapsed = time.time() - t0
        output = res.stdout + res.stderr

        # Count passed tests
        passed = 0
        if "passed" in output:
            import re
            m = re.search(r"(\d+) passed", output)
            if m:
                passed = int(m.group(1))

        if res.returncode == 0:
            passed_total += passed
            print(f"[OK]   {name:<42} {passed:>2} passed  ({elapsed:.2f}s)")
        else:
            failed_projects.append(name)
            print(f"[FAIL] {name:<42} failed")

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"Total passed tests: {passed_total} across {len(PROJECTS)} projects ({total_time:.2f}s)")

    if failed_projects:
        print(f"Failures in: {', '.join(failed_projects)}")
        sys.exit(1)
    else:
        print("All project tests passed successfully!")


if __name__ == "__main__":
    main()
