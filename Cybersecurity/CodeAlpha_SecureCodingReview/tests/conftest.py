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
