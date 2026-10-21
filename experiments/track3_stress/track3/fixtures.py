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
