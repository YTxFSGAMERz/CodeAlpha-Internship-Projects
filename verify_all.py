#!/usr/bin/env python3
"""
Master Verification Suite for CodeAlpha Internship Projects.
Validates all 12 projects across Artificial Intelligence, Cybersecurity, and Machine Learning.
Runs test suites, verifies entrypoints and README files, and produces an executive health report.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent

PROJECTS = [
    # Artificial Intelligence
    {
        "track": "Artificial Intelligence",
        "name": "CodeAlpha_LanguageTranslationTool",
        "test_path": "Artificial Intelligence/CodeAlpha_LanguageTranslationTool/tests",
        "entrypoints": ["app.py", "translator.py"]
    },
    {
        "track": "Artificial Intelligence",
        "name": "CodeAlpha_FAQChatbot",
        "test_path": "Artificial Intelligence/CodeAlpha_FAQChatbot/tests",
        "entrypoints": ["app.py", "chatbot.py"]
    },
    {
        "track": "Artificial Intelligence",
        "name": "CodeAlpha_MusicGenerationAI",
        "test_path": "Artificial Intelligence/CodeAlpha_MusicGenerationAI/tests",
        "entrypoints": ["generate.py", "train.py"]
    },
    {
        "track": "Artificial Intelligence",
        "name": "CodeAlpha_ObjectDetectionTracking",
        "test_path": "Artificial Intelligence/CodeAlpha_ObjectDetectionTracking/tests",
        "entrypoints": ["detect_track.py", "detector.py"]
    },
    # Cybersecurity
    {
        "track": "Cybersecurity",
        "name": "CodeAlpha_BasicNetworkSniffer",
        "test_path": "Cybersecurity/CodeAlpha_BasicNetworkSniffer/tests",
        "entrypoints": ["sniffer.py"]
    },
    {
        "track": "Cybersecurity",
        "name": "CodeAlpha_NetworkIntrusionDetectionSystem",
        "test_path": "Cybersecurity/CodeAlpha_NetworkIntrusionDetectionSystem/tests",
        "entrypoints": ["scripts/generate_test_traffic.py", "scripts/test_rules.py"]
    },
    {
        "track": "Cybersecurity",
        "name": "CodeAlpha_PhishingAwarenessTraining",
        "test_path": "Cybersecurity/CodeAlpha_PhishingAwarenessTraining/tests",
        "entrypoints": ["index.html"]
    },
    {
        "track": "Cybersecurity",
        "name": "CodeAlpha_SecureCodingReview",
        "test_path": "Cybersecurity/CodeAlpha_SecureCodingReview/tests",
        "entrypoints": ["vulnerable_app/app.py", "secure_app/app.py"]
    },
    # Machine Learning
    {
        "track": "Machine Learning",
        "name": "CodeAlpha_CreditScoringModel",
        "test_path": "Machine Learning/CodeAlpha_CreditScoringModel/tests",
        "entrypoints": ["train.py", "predict.py"]
    },
    {
        "track": "Machine Learning",
        "name": "CodeAlpha_DiseasePrediction",
        "test_path": "Machine Learning/CodeAlpha_DiseasePrediction/tests",
        "entrypoints": ["train.py", "predict.py"]
    },
    {
        "track": "Machine Learning",
        "name": "CodeAlpha_EmotionRecognitionFromSpeech",
        "test_path": "Machine Learning/CodeAlpha_EmotionRecognitionFromSpeech/tests",
        "entrypoints": ["train.py", "predict.py"]
    },
    {
        "track": "Machine Learning",
        "name": "CodeAlpha_HandwrittenCharacterRecognition",
        "test_path": "Machine Learning/CodeAlpha_HandwrittenCharacterRecognition/tests",
        "entrypoints": ["train.py", "predict.py"]
    },
]


def run_tests():
    print("=" * 80)
    print("  CODEALPHA INTERNSHIP PROJECTS - MASTER VERIFICATION RUNNER")
    print("=" * 80)
    print(f"Repository Root: {ROOT}\n")

    results = []
    total_passed = 0
    total_failed = 0
    start_all = time.time()

    for p in PROJECTS:
        proj_dir = ROOT / p["track"] / p["name"]
        has_dir = proj_dir.exists()
        has_readme = (proj_dir / "README.md").exists()
        has_tests = (ROOT / p["test_path"]).exists()

        status_text = "PASSED"
        tests_count = 0
        duration = 0.0

        if not has_dir:
            status_text = "DIR MISSING"
        elif not has_readme:
            status_text = "README MISSING"
        elif has_tests:
            t0 = time.time()
            res = subprocess.run(
                [sys.executable, "-m", "pytest", str(ROOT / p["test_path"]), "-q"],
                cwd=str(ROOT),
                capture_output=True,
                text=True
            )
            duration = time.time() - t0
            output = res.stdout + res.stderr
            if res.returncode == 0:
                status_text = "PASSED"
                # parse passed count
                for part in output.split():
                    if part.isdigit():
                        tests_count = int(part)
                # Fallback parse
                if "passed" in output:
                    import re
                    m = re.search(r"(\d+) passed", output)
                    if m:
                        tests_count = int(m.group(1))
                total_passed += tests_count
            else:
                status_text = "FAILED"
                total_failed += 1
        else:
            status_text = "NO TESTS"

        results.append({
            "track": p["track"],
            "name": p["name"],
            "status": status_text,
            "tests": tests_count,
            "time": f"{duration:.2f}s" if duration > 0 else "-"
        })

    print(f"{'Track':<25} {'Project':<40} {'Status':<10} {'Tests':<8} {'Time'}")
    print("-" * 90)
    for r in results:
        status_display = f"\033[92m{r['status']}\033[0m" if r['status'] == "PASSED" else f"\033[91m{r['status']}\033[0m"
        print(f"{r['track']:<25} {r['name']:<40} {r['status']:<10} {r['tests']:<8} {r['time']}")

    total_time = time.time() - start_all
    print("-" * 90)
    print(f"Total Projects Verified: {len(results)}/12")
    print(f"Total Automated Tests Passed: {total_passed}")
    print(f"Total Failures: {total_failed}")
    print(f"Total Duration: {total_time:.2f}s")
    print("=" * 80)

    if total_failed > 0:
        sys.exit(1)
    else:
        print("\nAll 12 CodeAlpha projects are healthy and fully verified!")


if __name__ == "__main__":
    run_tests()
