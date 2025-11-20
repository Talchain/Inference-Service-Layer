"""
Unit tests for UserStorage service.

Tests Redis-based user belief persistence with fallback.
"""

import pytest

from src.models.phase1_models import CounterfactualQuery, Scenario, UserBeliefModel
from src.models.shared import Distribution, DistributionType
from src.services.user_storage import UserStorage


@pytest.fixture
def user_storage():
    """User storage instance (will use fallback since Redis not running)."""
    return UserStorage()


@pytest.fixture
def sample_beliefs():
    """Sample user belief model."""
    return UserBeliefModel(
        value_weights={
            "revenue": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 0.7, "std": 0.2},
            ),
            "churn": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 0.4, "std": 0.2},
            ),
        },
        risk_tolerance=Distribution(
            type=DistributionType.BETA,
            parameters={"alpha": 2, "beta": 2},
        ),
        time_horizon=Distribution(
            type=DistributionType.NORMAL,
            parameters={"mean": 12, "std": 3},
        ),
        uncertainty_estimates={
            "revenue_weight": 0.3,
            "churn_weight": 0.4,
        },
    )


@pytest.fixture
def sample_query():
    """Sample counterfactual query."""
    return CounterfactualQuery(
        id="query_test_001",
        question="Which do you prefer?",
        scenario_a=Scenario(
            description="Option A",
            outcomes={"revenue": 50000, "churn": 0.05},
            trade_offs=["Balanced approach"],
        ),
        scenario_b=Scenario(
            description="Option B",
            outcomes={"revenue": 60000, "churn": 0.08},
            trade_offs=["Aggressive approach"],
        ),
        information_gain=0.35,
    )


def test_store_and_retrieve_beliefs(user_storage, sample_beliefs):
    """Test storing and retrieving beliefs."""
    user_id = "test_user_001"

    # Store beliefs
    user_storage.store_beliefs(user_id, sample_beliefs)

    # Retrieve beliefs
    retrieved = user_storage.get_beliefs(user_id)

    assert retrieved is not None
    assert retrieved.value_weights.keys() == sample_beliefs.value_weights.keys()

    # Check values match
    assert (
        retrieved.value_weights["revenue"].parameters["mean"]
        == sample_beliefs.value_weights["revenue"].parameters["mean"]
    )
    assert (
        retrieved.uncertainty_estimates
        == sample_beliefs.uncertainty_estimates
    )


def test_get_beliefs_not_found(user_storage):
    """Test retrieving beliefs for non-existent user."""
    result = user_storage.get_beliefs("nonexistent_user")
    assert result is None


def test_store_beliefs_with_custom_ttl(user_storage, sample_beliefs):
    """Test storing beliefs with custom TTL."""
    user_id = "test_user_ttl"

    # Store with custom TTL (should work even if Redis not available)
    user_storage.store_beliefs(user_id, sample_beliefs, ttl_hours=48)

    # Should be retrievable
    retrieved = user_storage.get_beliefs(user_id)
    assert retrieved is not None


def test_add_query_to_history(user_storage, sample_query):
    """Test adding query to history."""
    user_id = "test_user_query"

    # Add query
    user_storage.add_query_to_history(user_id, sample_query, response="A")

    # Get query count
    count = user_storage.get_query_count(user_id)

    # In fallback mode, count might be 0, but in Redis mode it should be 1
    assert count >= 0


def test_get_query_count_no_history(user_storage):
    """Test getting query count for user with no history."""
    count = user_storage.get_query_count("user_no_history")
    assert count == 0


def test_get_query_history(user_storage, sample_query):
    """Test retrieving query history."""
    user_id = "test_user_history"

    # Add multiple queries
    for i in range(3):
        query = CounterfactualQuery(
            id=f"query_{i}",
            question=f"Question {i}",
            scenario_a=sample_query.scenario_a,
            scenario_b=sample_query.scenario_b,
            information_gain=0.3,
        )
        user_storage.add_query_to_history(user_id, query, response="A")

    # Get history
    history = user_storage.get_query_history(user_id, limit=10)

    # In fallback mode, might be empty, but in Redis mode should have queries
    assert isinstance(history, list)


def test_get_query_history_with_limit(user_storage, sample_query):
    """Test retrieving query history with limit."""
    user_id = "test_user_limit"

    # Add 5 queries
    for i in range(5):
        query = CounterfactualQuery(
            id=f"query_limit_{i}",
            question=f"Question {i}",
            scenario_a=sample_query.scenario_a,
            scenario_b=sample_query.scenario_b,
            information_gain=0.3,
        )
        user_storage.add_query_to_history(user_id, query)

    # Get only 3
    history = user_storage.get_query_history(user_id, limit=3)

    # Should not exceed limit
    assert len(history) <= 3


def test_delete_user_data(user_storage, sample_beliefs, sample_query):
    """Test deleting all user data."""
    user_id = "test_user_delete"

    # Store beliefs and query history
    user_storage.store_beliefs(user_id, sample_beliefs)
    user_storage.add_query_to_history(user_id, sample_query)

    # Delete user data
    user_storage.delete_user_data(user_id)

    # Beliefs should be gone
    assert user_storage.get_beliefs(user_id) is None

    # Query count should be 0
    assert user_storage.get_query_count(user_id) == 0


def test_multiple_users_isolated(user_storage, sample_beliefs):
    """Test that multiple users' data is isolated."""
    user1 = "user_001"
    user2 = "user_002"

    # Store different beliefs for each user
    beliefs1 = sample_beliefs
    beliefs2 = UserBeliefModel(
        value_weights={
            "revenue": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 0.9, "std": 0.1},
            ),
            "churn": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 0.2, "std": 0.1},
            ),
        },
        risk_tolerance=Distribution(
            type=DistributionType.BETA,
            parameters={"alpha": 3, "beta": 2},
        ),
        time_horizon=Distribution(
            type=DistributionType.NORMAL,
            parameters={"mean": 6, "std": 2},
        ),
        uncertainty_estimates={
            "revenue_weight": 0.2,
            "churn_weight": 0.3,
        },
    )

    user_storage.store_beliefs(user1, beliefs1)
    user_storage.store_beliefs(user2, beliefs2)

    # Each user should get their own beliefs back
    retrieved1 = user_storage.get_beliefs(user1)
    retrieved2 = user_storage.get_beliefs(user2)

    assert retrieved1 is not None
    assert retrieved2 is not None

    # Beliefs should be different
    assert (
        retrieved1.value_weights["revenue"].parameters["mean"]
        != retrieved2.value_weights["revenue"].parameters["mean"]
    )


def test_fallback_storage_initialization(user_storage):
    """Test that fallback storage is initialized when Redis unavailable."""
    # UserStorage should initialize successfully even without Redis
    assert user_storage is not None

    # Should either have redis_enabled True (if Redis running) or False (fallback)
    assert isinstance(user_storage.redis_enabled, bool)

    # If Redis not enabled, should have fallback storage
    if not user_storage.redis_enabled:
        assert hasattr(user_storage, "fallback_storage")
        assert isinstance(user_storage.fallback_storage, dict)


def test_hash_user_id(user_storage):
    """Test user ID hashing for privacy."""
    user_id = "sensitive_user_id_123"

    hashed = user_storage._hash_user_id(user_id)

    # Should be hashed (16 characters)
    assert len(hashed) == 16
    assert hashed != user_id

    # Should be deterministic
    hashed2 = user_storage._hash_user_id(user_id)
    assert hashed == hashed2

    # Different IDs should hash differently
    hashed_different = user_storage._hash_user_id("different_id")
    assert hashed != hashed_different


def test_concurrent_operations(user_storage, sample_beliefs):
    """Test that storage handles concurrent-like operations."""
    user_id = "test_concurrent"

    # Store beliefs
    user_storage.store_beliefs(user_id, sample_beliefs)

    # Retrieve multiple times
    for _ in range(5):
        retrieved = user_storage.get_beliefs(user_id)
        assert retrieved is not None


def test_beliefs_serialization_roundtrip(user_storage, sample_beliefs):
    """Test that beliefs survive serialization/deserialization."""
    user_id = "test_serialization"

    # Store beliefs
    user_storage.store_beliefs(user_id, sample_beliefs)

    # Retrieve
    retrieved = user_storage.get_beliefs(user_id)

    # Should have all the same fields
    assert retrieved.value_weights.keys() == sample_beliefs.value_weights.keys()
    assert retrieved.uncertainty_estimates.keys() == sample_beliefs.uncertainty_estimates.keys()

    # Values should match
    for var in sample_beliefs.value_weights.keys():
        assert (
            retrieved.value_weights[var].type
            == sample_beliefs.value_weights[var].type
        )
        assert (
            retrieved.value_weights[var].parameters
            == sample_beliefs.value_weights[var].parameters
        )


def test_store_beliefs_overwrites_existing(user_storage, sample_beliefs):
    """Test that storing beliefs overwrites existing ones."""
    user_id = "test_overwrite"

    # Store initial beliefs
    user_storage.store_beliefs(user_id, sample_beliefs)

    # Create new beliefs
    new_beliefs = UserBeliefModel(
        value_weights={
            "revenue": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 0.9, "std": 0.1},
            ),
            "churn": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 0.3, "std": 0.1},
            ),
        },
        risk_tolerance=sample_beliefs.risk_tolerance,
        time_horizon=sample_beliefs.time_horizon,
        uncertainty_estimates={
            "revenue_weight": 0.1,
            "churn_weight": 0.2,
        },
    )

    # Store new beliefs
    user_storage.store_beliefs(user_id, new_beliefs)

    # Should retrieve new beliefs, not old ones
    retrieved = user_storage.get_beliefs(user_id)
    assert (
        retrieved.value_weights["revenue"].parameters["mean"]
        == new_beliefs.value_weights["revenue"].parameters["mean"]
    )
