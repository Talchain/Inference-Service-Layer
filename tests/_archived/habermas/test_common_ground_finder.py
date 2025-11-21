"""
Unit tests for CommonGroundFinder service.

Tests identification of shared values and concerns across team positions.
"""

import pytest

from src.models.deliberation import (
    MemberPosition,
    ValueStatement,
    ConcernStatement,
)
from src.services.common_ground_finder import CommonGroundFinder


class TestCommonGroundFinder:
    """Test common ground identification."""

    def setup_method(self):
        """Set up test fixtures."""
        self.finder = CommonGroundFinder(similarity_threshold=0.7)

    def test_empty_positions(self):
        """Test handling empty positions list."""
        common_ground = self.finder.find_common_ground(
            positions=[],
            request_id="test_001",
        )

        assert common_ground.agreement_level == 0.0
        assert len(common_ground.shared_values) == 0
        assert len(common_ground.participants) == 0

    def test_single_position(self):
        """Test handling single position."""
        position = MemberPosition(
            member_id="alice",
            values=[
                ValueStatement(
                    value_name="quality",
                    weight=0.8,
                    rationale="Quality matters",
                )
            ],
            timestamp="2025-01-01T00:00:00",
        )

        common_ground = self.finder.find_common_ground(
            positions=[position],
            request_id="test_002",
        )

        # Single position, no shared values (need 2+ members)
        assert len(common_ground.participants) == 1
        assert len(common_ground.shared_values) == 0

    def test_two_members_same_value(self):
        """Test identifying shared value between two members."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(
                        value_name="quality",
                        weight=0.8,
                        rationale="Quality is important",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(
                        value_name="quality",
                        weight=0.7,
                        rationale="Quality matters to me",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_003",
        )

        # Should find shared quality value
        assert len(common_ground.shared_values) > 0
        quality_shared = [v for v in common_ground.shared_values if "quality" in v.value_name]
        assert len(quality_shared) > 0

        # Check agreement score
        shared_quality = quality_shared[0]
        assert shared_quality.agreement_score == 1.0  # 2/2 = 100%
        assert len(shared_quality.supporting_members) == 2

    def test_average_weight_calculation(self):
        """Test that average weight is calculated correctly."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(
                        value_name="speed",
                        weight=0.8,
                        rationale="Speed matters",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(
                        value_name="speed",
                        weight=0.6,
                        rationale="Speed is good",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_004",
        )

        speed_shared = [v for v in common_ground.shared_values if "speed" in v.value_name]
        assert len(speed_shared) > 0

        # Average of 0.8 and 0.6 should be 0.7
        assert speed_shared[0].average_weight == pytest.approx(0.7, abs=0.01)

    def test_multiple_shared_values(self):
        """Test identifying multiple shared values."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(value_name="quality", weight=0.8, rationale="test"),
                    ValueStatement(value_name="speed", weight=0.7, rationale="test"),
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(value_name="quality", weight=0.9, rationale="test"),
                    ValueStatement(value_name="speed", weight=0.6, rationale="test"),
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_005",
        )

        # Should find both shared values
        assert len(common_ground.shared_values) >= 2

    def test_partial_agreement(self):
        """Test partial agreement (some share, some don't)."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(value_name="quality", weight=0.8, rationale="test")
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(value_name="quality", weight=0.7, rationale="test")
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="charlie",
                values=[
                    ValueStatement(value_name="speed", weight=0.9, rationale="test")
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_006",
        )

        # Should find quality as shared (2/3 = 0.67 agreement)
        quality_shared = [v for v in common_ground.shared_values if "quality" in v.value_name]
        assert len(quality_shared) > 0
        assert quality_shared[0].agreement_score == pytest.approx(2 / 3, abs=0.01)

    def test_shared_concerns_identification(self):
        """Test identifying shared concerns."""
        positions = [
            MemberPosition(
                member_id="alice",
                concerns=[
                    ConcernStatement(
                        concern_name="technical_risk",
                        severity=0.8,
                        explanation="Scalability issues",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                concerns=[
                    ConcernStatement(
                        concern_name="technical_risk",
                        severity=0.7,
                        explanation="Performance problems",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_007",
        )

        # Should find shared concerns
        assert len(common_ground.shared_concerns) > 0

        tech_concern = [c for c in common_ground.shared_concerns if "technical" in c.value_name]
        assert len(tech_concern) > 0

    def test_agreement_level_high_alignment(self):
        """Test agreement level with high alignment."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(value_name="quality", weight=0.9, rationale="test"),
                    ValueStatement(value_name="speed", weight=0.8, rationale="test"),
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(value_name="quality", weight=0.9, rationale="test"),
                    ValueStatement(value_name="speed", weight=0.7, rationale="test"),
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_008",
        )

        # High alignment should yield high agreement level
        assert common_ground.agreement_level > 0.5

    def test_agreement_level_low_alignment(self):
        """Test agreement level with low alignment."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(value_name="quality", weight=0.9, rationale="test")
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(value_name="speed", weight=0.9, rationale="test")
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_009",
        )

        # No shared values, should have low/zero agreement
        assert common_ground.agreement_level == 0.0

    def test_participants_list_accurate(self):
        """Test that participants list is accurate."""
        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="bob", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="charlie", timestamp="2025-01-01T00:00:00"),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_010",
        )

        assert len(common_ground.participants) == 3
        assert "alice" in common_ground.participants
        assert "bob" in common_ground.participants
        assert "charlie" in common_ground.participants

    def test_synthesized_rationale_generated(self):
        """Test that synthesized rationale is generated."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(
                        value_name="quality",
                        weight=0.8,
                        rationale="Quality ensures reliability",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(
                        value_name="quality",
                        weight=0.7,
                        rationale="Quality reduces bugs",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_011",
        )

        quality_shared = [v for v in common_ground.shared_values if "quality" in v.value_name]
        assert len(quality_shared) > 0

        # Should have synthesized rationale
        assert len(quality_shared[0].synthesized_rationale) > 0

    def test_sorting_by_agreement_score(self):
        """Test that shared values are sorted by agreement score."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(value_name="quality", weight=0.8, rationale="test"),
                    ValueStatement(value_name="speed", weight=0.7, rationale="test"),
                    ValueStatement(value_name="cost", weight=0.6, rationale="test"),
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(value_name="quality", weight=0.9, rationale="test"),
                    ValueStatement(value_name="speed", weight=0.8, rationale="test"),
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="charlie",
                values=[
                    ValueStatement(value_name="quality", weight=0.7, rationale="test")
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_012",
        )

        # quality: 3/3 = 1.0
        # speed: 2/3 = 0.67
        # Should be sorted descending by agreement
        if len(common_ground.shared_values) >= 2:
            scores = [v.agreement_score for v in common_ground.shared_values]
            assert scores == sorted(scores, reverse=True)

    def test_normalization_ignores_case(self):
        """Test that value name normalization ignores case."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(value_name="Quality", weight=0.8, rationale="test")
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(value_name="quality", weight=0.7, rationale="test")
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_013",
        )

        # Should recognize as same value despite case difference
        assert len(common_ground.shared_values) > 0

    def test_normalization_handles_underscores(self):
        """Test that normalization handles underscores."""
        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(
                        value_name="user_satisfaction",
                        weight=0.8,
                        rationale="test",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(
                        value_name="user satisfaction",
                        weight=0.7,
                        rationale="test",
                    )
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        common_ground = self.finder.find_common_ground(
            positions=positions,
            request_id="test_014",
        )

        # Should recognize as same value despite underscore/space difference
        assert len(common_ground.shared_values) > 0
