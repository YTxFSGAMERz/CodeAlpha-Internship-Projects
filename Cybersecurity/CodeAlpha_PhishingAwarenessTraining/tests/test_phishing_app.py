"""
Automated validation of the Phishing Awareness Training web application.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HTML_FILE = ROOT / "index.html"


def test_html_exists():
    assert HTML_FILE.exists(), "index.html must exist"


def test_html_contains_training_modules():
    content = HTML_FILE.read_text(encoding="utf-8")
    
    # Must contain HTML5 doctype and title
    assert "<!doctype html>" in content.lower()
    assert "<title>Phishing Awareness Training</title>" in content
    
    # Check for core sections
    assert "Urgent request for action" in content or "phishing" in content.lower()
    assert "interactive" in content.lower() or "quiz" in content.lower() or "indicator" in content.lower()
    
    # Check for interactive elements
    assert "<button" in content
    assert "<style>" in content
