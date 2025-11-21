"""
Unit tests for ValueExtractor service.

Tests extraction of values and concerns from free-form text.
"""

import pytest

from src.services.value_extractor import ValueExtractor


class TestValueExtractor:
    """Test value extraction from position statements."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ValueExtractor()

    def test_extract_user_value(self):
        """Test extracting user-focused values."""
        text = "User satisfaction is critical. We must prioritize customer needs."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="alice",
            context="Feature decision",
            request_id="test_001",
        )

        assert position.member_id == "alice"
        assert len(position.values) > 0

        # Should find user-related value
        user_values = [v for v in position.values if "user" in v.value_name]
        assert len(user_values) > 0

    def test_extract_quality_value(self):
        """Test extracting quality-focused values."""
        text = "Quality and reliability are essential. We need robust, stable systems."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="bob",
            context="Architecture decision",
            request_id="test_002",
        )

        # Should find quality value
        quality_values = [v for v in position.values if "quality" in v.value_name]
        assert len(quality_values) > 0

    def test_extract_speed_value(self):
        """Test extracting speed/velocity values."""
        text = "Speed is important. We need fast time-to-market and rapid delivery."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="charlie",
            context="Timeline decision",
            request_id="test_003",
        )

        # Should find speed value
        speed_values = [v for v in position.values if "speed" in v.value_name]
        assert len(speed_values) > 0

    def test_weight_calculation_with_emphasis(self):
        """Test that emphasis words increase weight."""
        text_critical = "User satisfaction is critical and essential for success."
        text_normal = "User satisfaction matters to us."

        pos_critical = self.extractor.extract_values_and_concerns(
            position_text=text_critical,
            member_id="alice",
            context="test",
            request_id="test_004",
        )

        pos_normal = self.extractor.extract_values_and_concerns(
            position_text=text_normal,
            member_id="bob",
            context="test",
            request_id="test_005",
        )

        # Critical emphasis should yield higher weight
        if pos_critical.values and pos_normal.values:
            critical_weight = max(v.weight for v in pos_critical.values)
            normal_weight = max(v.weight for v in pos_normal.values)
            assert critical_weight >= normal_weight

    def test_extract_concerns(self):
        """Test extracting concerns from text."""
        text = "I'm worried about technical risks. The system might not scale properly."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="alice",
            context="Scalability discussion",
            request_id="test_006",
        )

        # Should extract concerns
        assert len(position.concerns) > 0

        # Should identify as technical risk
        technical_concerns = [
            c for c in position.concerns if "technical" in c.concern_name
        ]
        assert len(technical_concerns) > 0

    def test_concern_severity_calculation(self):
        """Test that severity indicators are detected."""
        text_critical = "I have a critical concern about major technical risks."
        text_minor = "There's a minor issue with the approach."

        pos_critical = self.extractor.extract_values_and_concerns(
            position_text=text_critical,
            member_id="alice",
            context="test",
            request_id="test_007",
        )

        pos_minor = self.extractor.extract_values_and_concerns(
            position_text=text_minor,
            member_id="bob",
            context="test",
            request_id="test_008",
        )

        # Critical should have higher severity
        if pos_critical.concerns and pos_minor.concerns:
            critical_severity = pos_critical.concerns[0].severity
            minor_severity = pos_minor.concerns[0].severity
            assert critical_severity > minor_severity

    def test_extract_examples(self):
        """Test extracting examples from text."""
        text = (
            "User satisfaction is important. "
            "For example, fast response times and intuitive interfaces matter."
        )

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="alice",
            context="UX decision",
            request_id="test_009",
        )

        # Should extract examples
        values_with_examples = [v for v in position.values if v.examples]
        assert len(values_with_examples) > 0

    def test_multiple_values_in_text(self):
        """Test extracting multiple different values."""
        text = (
            "User satisfaction and technical quality are both critical. "
            "We also need fast delivery and innovation."
        )

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="alice",
            context="Multi-value decision",
            request_id="test_010",
        )

        # Should extract multiple values
        assert len(position.values) >= 2

    def test_empty_text_handling(self):
        """Test handling empty or minimal text."""
        text = ""

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="alice",
            context="test",
            request_id="test_011",
        )

        # Should still create position with default value
        assert position.member_id == "alice"
        assert len(position.values) > 0  # Default value

    def test_position_statement_preserved(self):
        """Test that original position statement is preserved."""
        text = "This is my complete position on the matter."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="alice",
            context="test",
            request_id="test_012",
        )

        assert position.position_statement == text

    def test_timestamp_generated(self):
        """Test that timestamp is generated."""
        text = "User satisfaction matters."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="alice",
            context="test",
            request_id="test_013",
        )

        assert position.timestamp is not None
        assert len(position.timestamp) > 0

    def test_schedule_concern_identification(self):
        """Test identifying schedule/timeline concerns."""
        text = "I'm worried about the deadline. We might be late on delivery."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="alice",
            context="Timeline",
            request_id="test_014",
        )

        # Should identify schedule risk
        schedule_concerns = [c for c in position.concerns if "schedule" in c.concern_name]
        assert len(schedule_concerns) > 0

    def test_business_value_extraction(self):
        """Test extracting business-focused values."""
        text = "Revenue growth and market expansion are critical business priorities."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="ceo",
            context="Business strategy",
            request_id="test_015",
        )

        # Should find business value
        business_values = [v for v in position.values if "business" in v.value_name]
        assert len(business_values) > 0

    def test_team_value_extraction(self):
        """Test extracting team/collaboration values."""
        text = "Team collaboration and good communication are essential for success."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="manager",
            context="Process decision",
            request_id="test_016",
        )

        # Should find team value
        team_values = [v for v in position.values if "team" in v.value_name]
        assert len(team_values) > 0

    def test_rationale_extraction(self):
        """Test that rationale is meaningful."""
        text = "Quality is critical. We must ensure robust, reliable systems that users trust."

        position = self.extractor.extract_values_and_concerns(
            position_text=text,
            member_id="alice",
            context="Quality discussion",
            request_id="test_017",
        )

        # Rationale should contain relevant content
        for value in position.values:
            assert len(value.rationale) > 0
            # Should not be just a default placeholder
            assert value.rationale != "Values unknown"
