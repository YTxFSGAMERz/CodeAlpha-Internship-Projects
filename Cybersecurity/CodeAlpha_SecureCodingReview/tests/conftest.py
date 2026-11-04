"""Collects the proof-of-concept results and prints one summary table."""
RESULTS = {}


def pytest_terminal_summary(terminalreporter):
    if not RESULTS:
        return
    tr = terminalreporter
    tr.section("Proof-of-concept summary")
    tr.write_line(f"{'Finding':<46}{'vulnerable_app':<17}secure_app")
    for (fid, title), r in sorted(RESULTS.items()):
        tr.write_line(f"{fid + ' ' + title:<46}"
                      f"{'EXPLOITED' if r.get('vulnerable') else 'not exploited':<17}"
                      f"{'EXPLOITED' if r.get('secure') else 'blocked'}")


if __name__ == "__main__":
    import os
    import sys
    import pytest
    test_file = os.path.join(os.path.dirname(__file__), "test_findings.py")
    print("Executing security proof-of-concept tests...\n")
    sys.exit(pytest.main(["-v", test_file]))
