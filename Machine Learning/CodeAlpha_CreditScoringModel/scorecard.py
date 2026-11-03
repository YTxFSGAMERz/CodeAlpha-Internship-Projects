"""Turn a probability of default into a familiar 300-850 credit score."""
import numpy as np

# Illustrative scorecard scaling (the standard "points to double the odds" method):
#   600 points  <->  30 : 1 odds of repaying vs defaulting (about 3% default probability)
#   every +40 points doubles the odds of repaying
BASE_SCORE, BASE_ODDS, PDO = 600, 30, 40


def probability_to_score(p_default, base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO,
                         low=300, high=850):
    p = np.clip(np.asarray(p_default, dtype=float), 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    score = offset + factor * np.log((1 - p) / p)
    return np.clip(np.round(score), low, high).astype(int)


def decision(p_default, threshold):
    """Simple three-way policy built on the model's tuned threshold."""
    p = np.asarray(p_default, dtype=float)
    return np.select([p >= threshold, p >= 0.5 * threshold],
                     ["DECLINE", "MANUAL REVIEW"], default="APPROVE")
