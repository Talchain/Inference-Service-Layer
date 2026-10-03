"""
M2 "why it moved" (CEE #2410, PLoT #430): the invariant C1 attribution rests on.

CEE may say a rerun's movement was CAUSED by the user's edit (``C1_attributable``) only when both Runs drew their
samples the same way. With the same seed (CEE lends the prior Run's seed), that holds on this service iff an edit to
one edge's strength ``mean`` / ``std`` leaves every OTHER draw unchanged. It holds today because the edge strength is
one unbounded ``rng.normal(mean, std)`` per existing edge (``_sample_edge_strength``): the draw count never depends
on mean/std. A rejection sampler (``SeededRNG.truncated_normal``) would break it: its draw count depends on how much
mass falls outside the bounds, so a mean edit shifts every later draw (PLoT #430 overflow P1, read on a prototype
branch; SCIENCE/DSK 5935983506).

Pinned:
  S1  ``rng.normal`` advances the bit generator identically whatever (mean, std).
  S3  same seed, one edge's mean (or std) edited -> every other edge's strengths identical over all N iterations.
  S4  (documents) an ``exists_probability`` edit DOES move later draws (the strength is drawn only when the edge
      exists), so CEE's draw-structure key must keep classifying it unpaired (C2). Not a defect here.
  S5  (control) adding a link moves later draws (structure change -> C2).
Mutant: make ``_sample_edge_strength`` call ``rng.truncated_normal(mean, std)`` -> S3 is RED.
"""

import copy

import pytest

from src.models.robustness_v2 import EdgeV2
from src.services.robustness_analyzer_v2 import DualUncertaintySampler
from src.utils.rng import SeededRNG

N = 5000
SEED = 777
BASE = [
    ("a", "y", 0.9, 0.5, 0.1),
    ("b", "y", 0.8, 0.3, 0.2),
    ("c", "y", 1.0, -0.4, 0.15),
    ("d", "y", 0.7, 0.2, 0.05),
]
OTHERS = [("b", "y"), ("c", "y"), ("d", "y")]


def _edges(spec):
    return [
        EdgeV2(**{"from": f, "to": t, "exists_probability": p, "strength": {"mean": m, "std": s}})
        for (f, t, p, m, s) in spec
    ]


def _draws(spec, seed=SEED):
    return DualUncertaintySampler(_edges(spec), SeededRNG(seed)).sample_n_configurations(N)


def _mismatches(base, other, keys):
    return sum(1 for i in range(N) for k in keys if base[i][k] != other[i][k])


def _with_edge_a(p, mean, std):
    spec = copy.deepcopy(BASE)
    spec[0] = ("a", "y", p, mean, std)
    return spec


@pytest.mark.parametrize("mean,std", [(0.5, 0.1), (0.99, 0.1), (-0.99, 0.2), (0.0, 1.0)])
def test_s1_normal_advances_the_generator_identically_for_any_mean_and_std(mean, std):
    reference = SeededRNG(SEED)
    reference.normal(0.0, 1.0)
    rng = SeededRNG(SEED)
    rng.normal(mean, std)
    assert rng._rng.bit_generator.state == reference._rng.bit_generator.state


@pytest.mark.parametrize(
    "mean,std",
    [(0.99, 0.1), (-0.99, 0.2), (0.0, 1.0), (0.5, 0.4)],
    ids=["mean .5->.99", "mean ->-.99 std ->.2", "mean ->0 std ->1", "std .1->.4"],
)
def test_s3_a_strength_edit_leaves_every_other_edge_draw_identical(mean, std):
    base = _draws(BASE)
    edited = _draws(_with_edge_a(0.9, mean, std))
    assert _mismatches(base, edited, OTHERS) == 0
    # The edited edge itself moves (positive control: the edit reached the sampler).
    assert _mismatches(base, edited, [("a", "y")]) > 0


def test_s4_an_existence_edit_moves_later_draws_so_cee_must_call_it_unpaired():
    base = _draws(BASE)
    edited = _draws(_with_edge_a(0.5, 0.5, 0.1))
    assert _mismatches(base, edited, OTHERS) > N  # far from paired: comparable to a different seed


def test_s5_control_adding_a_link_moves_later_draws():
    base = _draws(BASE)
    added = _draws(copy.deepcopy(BASE) + [("e", "y", 1.0, 0.1, 0.1)])
    assert _mismatches(base, added, OTHERS) > N


def test_control_the_probe_sees_a_different_seed():
    assert _mismatches(_draws(BASE), _draws(BASE, seed=SEED + 1), OTHERS) > N
