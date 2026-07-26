"""
Pytest configuration and fixtures.

Provides reusable test fixtures for all test modules.
Includes quarantine infrastructure for pre-existing failing tests (H.6).
"""

import pathlib

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.api.main import app


# ---------------------------------------------------------------------------
# Quarantine infrastructure (H.6)
# ---------------------------------------------------------------------------
_QUARANTINE_FILE = pathlib.Path(__file__).parent / "_quarantined" / "known_failures.txt"


def _load_quarantined_ids():
    """Load quarantined test node IDs from the known-failures list."""
    if not _QUARANTINE_FILE.exists():
        return set()
    ids = set()
    for line in _QUARANTINE_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


_QUARANTINED_IDS = _load_quarantined_ids()


def pytest_addoption(parser):
    """Register custom CLI options."""
    parser.addoption(
        "--run-quarantined",
        action="store_true",
        default=False,
        help="Include quarantined tests in the run (they are skipped by default).",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark quarantined tests and skip them unless explicitly requested.

    Default: quarantined tests are skipped.
    To run ONLY quarantined tests: pytest -m quarantined
    To run ALL tests including quarantined: pytest --run-quarantined
    """
    # Check if user explicitly requested quarantined tests
    marker_expr = config.getoption("-m", default="")
    run_quarantined = (
        "quarantined" in str(marker_expr)
        or config.getoption("--run-quarantined", default=False)
    )

    quarantine_marker = pytest.mark.quarantined
    skip_marker = pytest.mark.skip(
        reason="Quarantined: pre-existing failure (see tests/_quarantined/MANIFEST.md)"
    )

    for item in items:
        if item.nodeid in _QUARANTINED_IDS:
            item.add_marker(quarantine_marker)
            if not run_quarantined:
                item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Auto-scaled noise opt-in (arch step 1, 2026-07-26)
# ---------------------------------------------------------------------------


@pytest.fixture
def auto_noise_enabled(monkeypatch):
    """Turn the auto-scaled noise heuristic ON for tests that pin its behaviour.

    `ENABLE_AUTO_SCALED_NOISE` defaults to False: the ~sqrt(2) spread inflation is
    an uncalibrated PoC heuristic by its own docstring's admission, and a client
    had no way to decline it, so it is opt-in until calibrated.

    Tests that pin NOISED numbers (wire value pins, seeded goldens, byte-identity
    checks) use this fixture rather than having their baselines re-derived. Two
    reasons: the pinned values stay exactly as they were, so this change cannot
    hide a regression behind a re-baselining; and the fixture makes each such
    test SAY that its subject is the noise-on path, which the tests could not say
    before because there was no other path.

    Tests whose subject is the DEFAULT served behaviour must not use it.
    """
    from src.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("ENABLE_AUTO_SCALED_NOISE", "true")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest_asyncio.fixture
async def client():
    """
    Async HTTP client for FastAPI testing.

    Uses httpx.AsyncClient instead of TestClient to avoid known Starlette
    async middleware bugs (anyio.EndOfStream errors).

    Returns:
        AsyncClient: Async HTTP client for making requests
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def sample_dag():
    """
    Simple causal DAG for testing.

    Structure: Z → X → Y, Z → Y (Z confounds X-Y relationship)

    Returns:
        dict: DAG structure
    """
    return {
        "nodes": ["X", "Y", "Z"],
        "edges": [["Z", "X"], ["X", "Y"], ["Z", "Y"]],
    }


@pytest.fixture
def pricing_dag():
    """
    Realistic pricing decision DAG.

    Returns:
        dict: Pricing DAG structure
    """
    return {
        "nodes": ["Price", "Brand", "Revenue", "CustomerAcquisition"],
        "edges": [
            ["Price", "Revenue"],
            ["Brand", "Price"],
            ["Brand", "Revenue"],
            ["CustomerAcquisition", "Revenue"],
        ],
    }


@pytest.fixture
def sample_structural_model():
    """
    Simple structural model for testing.

    Returns:
        dict: Structural model
    """
    return {
        "variables": ["X", "Y", "Z"],
        "equations": {"Y": "10 + 2*X + 3*Z", "Z": "5 + 0.5*X"},
        "distributions": {
            "X": {"type": "normal", "parameters": {"mean": 0, "std": 1}}
        },
    }


@pytest.fixture
def pricing_structural_model():
    """
    Realistic pricing structural model.

    Returns:
        dict: Pricing structural model
    """
    return {
        "variables": ["Price", "Brand", "Revenue"],
        "equations": {
            "Brand": "baseline_brand + 0.3 * Price",
            "Revenue": "10000 + 500 * Price - 200 * Brand",
        },
        "distributions": {
            "baseline_brand": {"type": "normal", "parameters": {"mean": 50, "std": 5}}
        },
    }


@pytest.fixture
def team_perspectives():
    """
    Sample team perspectives for testing.

    Returns:
        list: Team perspectives
    """
    return [
        {
            "role": "Product Manager",
            "priorities": ["User acquisition", "Revenue growth", "Fast time-to-market"],
            "constraints": ["Limited budget", "Q4 deadline"],
            "preferred_options": ["option_a", "option_b"],
        },
        {
            "role": "Designer",
            "priorities": ["User experience", "Brand consistency", "Accessibility"],
            "constraints": ["Design system limitations"],
            "preferred_options": ["option_b", "option_c"],
        },
        {
            "role": "Engineer",
            "priorities": ["Code quality", "Maintainability", "Tech debt reduction"],
            "constraints": ["Team capacity", "Technical limitations"],
            "preferred_options": ["option_c"],
        },
    ]


@pytest.fixture
def decision_options():
    """
    Sample decision options for testing.

    Returns:
        list: Decision options
    """
    return [
        {
            "id": "option_a",
            "name": "Quick MVP launch",
            "attributes": {
                "speed": "fast",
                "quality": "medium",
                "acquisition_potential": "high",
            },
        },
        {
            "id": "option_b",
            "name": "Polished feature set",
            "attributes": {
                "speed": "medium",
                "quality": "high",
                "acquisition_potential": "medium",
            },
        },
        {
            "id": "option_c",
            "name": "Architectural refactor first",
            "attributes": {
                "speed": "slow",
                "quality": "high",
                "acquisition_potential": "low",
            },
        },
    ]


@pytest.fixture
def sample_assumptions():
    """
    Sample assumptions for sensitivity analysis.

    Returns:
        list: Assumptions
    """
    return [
        {
            "name": "Customer price sensitivity",
            "current_value": 0.5,
            "type": "parametric",
            "variation_range": {"min": 0.3, "max": 0.8},
        },
        {
            "name": "Competitor response timing",
            "current_value": "90 days",
            "type": "structural",
        },
        {
            "name": "Seasonal demand variation",
            "current_value": {"Q4_multiplier": 1.08},
            "type": "distributional",
        },
    ]
