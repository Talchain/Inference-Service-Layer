"""Brief v1.0 §6 G1/G2 fixtures and an independent exact oracle.

The oracle reads only the brief's equations, in exact rational arithmetic. It never
reads the engine graph the scenario builder makes, so a builder or adapter fault
cannot also move the reference.
"""
from fractions import Fraction as Q

UNIT = "£000"
HORIZON_MONTHS = 12
TARGET = Q(90)            # Y >= 90, inclusive ("equality meets")
TARGET_OPERATOR = ">="

# option id -> (label, contribution with no stress, loss per unit demand stress x, loss per unit cost stress y)
OPTIONS = {
    "A": ("Growth-first", Q(120), Q(20), Q(20)),
    "B": ("Balanced rollout", Q(105), Q(10), Q(5)),
    "C": ("Protect margin", Q(95), Q(2), Q(2)),
}

HYPOTHESES = {
    "M0": {"label": "No retention-loss mechanism", "retention_loss": {}},
    "M1": {"label": "Growth-first activates a persistent retention loss",
           "retention_loss": {"A": Q(30)},
           "origin": "Brief v1.0 §6 G2: illustrative £30k assumption, not a measured effect"},
}

# Dimensionless synthetic stress coordinates (brief §6), never percentages from language.
G2_SETTINGS = [(Q(0), Q(0)), (Q(0), Q(1)), (Q(1), Q(0)), (Q(1), Q(1))]
# G1 = option A under M0: the four corners above plus the boundary point and one point either side.
G1_EXTRA = [(Q("0.75"), Q("0.75")), (Q("0.70"), Q("0.70")), (Q("0.80"), Q("0.80"))]


def oracle_level(option, hypothesis, x, y):
    _, base, dx, dy = OPTIONS[option]
    return base - dx * Q(x) - dy * Q(y) - HYPOTHESES[hypothesis]["retention_loss"].get(option, Q(0))


def meets(level, operator=TARGET_OPERATOR, target=TARGET):
    return level >= target if operator == ">=" else level > target


def scenario_id(hypothesis, x, y):
    return f"{hypothesis}:x={float(x):g},y={float(y):g}"


# ── G3 (brief §6): two target indicators, both marginal 0.9, three EXPLICIT joint laws.
G3_MARGINAL = Q(9, 10)
G3_LAWS = {
    "independent":       {(1, 1): Q(81, 100), (1, 0): Q(9, 100), (0, 1): Q(9, 100), (0, 0): Q(1, 100)},
    "always_together":   {(1, 1): Q(9, 10),   (1, 0): Q(0),      (0, 1): Q(0),      (0, 0): Q(1, 10)},
    "disjoint_failures": {(1, 1): Q(8, 10),   (1, 0): Q(1, 10),  (0, 1): Q(1, 10),  (0, 0): Q(0)},
}
# A valid table with the same marginals but no engine mapping here: it must come back unsupported, unevaluated.
G3_UNMAPPED = {"partial_overlap": {(1, 1): Q(85, 100), (1, 0): Q(5, 100), (0, 1): Q(5, 100), (0, 0): Q(5, 100)}}
G3_INVALID = {"sums_to_1.1": {(1, 1): Q(85, 100), (1, 0): Q(1, 10), (0, 1): Q(1, 10), (0, 0): Q(5, 100)},
              "negative_mass": {(1, 1): Q(95, 100), (1, 0): Q(-5, 100), (0, 1): Q(5, 100), (0, 0): Q(5, 100)},
              "wrong_marginal": {(1, 1): Q(7, 10), (1, 0): Q(1, 10), (0, 1): Q(2, 10), (0, 0): Q(0)}}


def admit_joint(table, stated_marginals=None):
    """Admit an explicit joint mass table, or refuse it with the reason. Never repairs a table.
    With stated marginals, a table that contradicts them is refused too."""
    if set(table) != {(1, 1), (1, 0), (0, 1), (0, 0)}:
        return {"admitted": False, "reason": "not a complete 2x2 mass table"}
    if any(m < 0 for m in table.values()):
        return {"admitted": False, "reason": "negative mass"}
    if sum(table.values()) != 1:
        return {"admitted": False, "reason": f"masses sum to {sum(table.values())}, not 1"}
    p1, p2 = table[(1, 1)] + table[(1, 0)], table[(1, 1)] + table[(0, 1)]
    if stated_marginals is not None and (p1, p2) != tuple(stated_marginals):
        return {"admitted": False, "reason": f"marginals {p1}, {p2} contradict the stated {stated_marginals[0]}, {stated_marginals[1]}"}
    return {"admitted": True, "p1": p1, "p2": p2, "p_both": table[(1, 1)]}


def joint_from_marginals_only(p1, p2):
    """Marginals alone do not identify the joint: return the Fréchet bounds, never a point."""
    return {"identified": False, "bounds": (max(Q(0), p1 + p2 - 1), min(p1, p2))}


# ── G4 (brief §6): Y ~ Normal(mu, 10^2) in £000, target Y >= 100.
G4_SD, G4_TARGET, G4_MUS = Q(10), Q(100), (Q(90), Q(100), Q(110))


def normal_goal_probability(mu, sd=G4_SD, target=G4_TARGET):
    import math
    return 0.5 * math.erfc(float(target - mu) / (float(sd) * math.sqrt(2)))
