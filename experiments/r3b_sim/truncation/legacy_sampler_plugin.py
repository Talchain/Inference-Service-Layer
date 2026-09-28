import pytest
@pytest.fixture(autouse=True)
def _legacy_sampler(monkeypatch):
    from src.services import robustness_analyzer_v2 as rav2
    monkeypatch.setattr(rav2, "_sample_edge_strength", lambda rng, m, s: rng.truncated_normal(m, s, -1.0, 1.0))
