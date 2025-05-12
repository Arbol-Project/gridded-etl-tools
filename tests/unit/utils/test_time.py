import pytest
from gridded_etl_tools.utils.time import TimeUnit, TimeSpan


class TestTimeUnit:
    """Test suite for TimeUnit class."""

    def test_valid_initialization(self):
        """Test that TimeUnit can be initialized with valid values."""
        unit = TimeUnit("hours", 1)
        assert unit.unit == "hours"
        assert unit.value == 1

    def test_invalid_unit(self):
        """Test that TimeUnit raises ValueError for invalid units."""
        with pytest.raises(ValueError, match="Invalid time unit: invalid_unit"):
            TimeUnit("invalid_unit", 1)

    def test_negative_value(self):
        """Test that TimeUnit raises ValueError for negative values."""
        with pytest.raises(ValueError, match="Time unit value must be positive"):
            TimeUnit("hours", -1)

    def test_zero_value(self):
        """Test that TimeUnit raises ValueError for zero values."""
        with pytest.raises(ValueError, match="Time unit value must be positive"):
            TimeUnit("hours", 0)

    @pytest.mark.parametrize(
        "unit,value,expected_minutes",
        [
            ("minutes", 1, 1),
            ("hours", 1, 60),
            ("days", 1, 24 * 60),
            ("weeks", 1, 7 * 24 * 60),
            ("months", 1, 30 * 24 * 60),  # Approximation
            ("years", 1, 365 * 24 * 60),  # Approximation
            ("seasons", 1, 90 * 24 * 60),  # Approximation
            ("minutes", 30, 30),
            ("hours", 2, 120),
            ("days", 2, 48 * 60),
        ],
    )
    def test_to_minutes(self, unit, value, expected_minutes):
        """Test conversion of various time units to minutes."""
        time_unit = TimeUnit(unit, value)
        assert time_unit.to_minutes() == expected_minutes

    def test_str_representation(self):
        """Test string representation of TimeUnit."""
        unit = TimeUnit("hours", 2)
        assert str(unit) == "2 hours"


class TestTimeSpan:
    """Test suite for TimeSpan enum."""

    def test_enum_members(self):
        """Test that all expected TimeSpan members exist with correct values."""
        assert TimeSpan.SPAN_HALF_HOURLY.value == TimeUnit("minutes", 30)
        assert TimeSpan.SPAN_HOURLY.value == TimeUnit("hours", 1)
        assert TimeSpan.SPAN_THREE_HOURLY.value == TimeUnit("hours", 3)
        assert TimeSpan.SPAN_SIX_HOURLY.value == TimeUnit("hours", 6)
        assert TimeSpan.SPAN_DAILY.value == TimeUnit("days", 1)
        assert TimeSpan.SPAN_WEEKLY.value == TimeUnit("weeks", 1)
        assert TimeSpan.SPAN_MONTHLY.value == TimeUnit("months", 1)
        assert TimeSpan.SPAN_YEARLY.value == TimeUnit("years", 1)
        assert TimeSpan.SPAN_SEASONAL.value == TimeUnit("seasons", 1)

    @pytest.mark.parametrize(
        "span_str,expected_span",
        [
            ("half_hourly", TimeSpan.SPAN_HALF_HOURLY),
            ("hourly", TimeSpan.SPAN_HOURLY),
            ("3hourly", TimeSpan.SPAN_THREE_HOURLY),
            ("6hourly", TimeSpan.SPAN_SIX_HOURLY),
            ("daily", TimeSpan.SPAN_DAILY),
            ("weekly", TimeSpan.SPAN_WEEKLY),
            ("monthly", TimeSpan.SPAN_MONTHLY),
            ("yearly", TimeSpan.SPAN_YEARLY),
            ("seasonal", TimeSpan.SPAN_SEASONAL),
        ],
    )
    def test_from_string_valid(self, span_str, expected_span):
        """Test conversion from valid string representations."""
        assert TimeSpan.from_string(span_str) == expected_span

    def test_from_string_invalid(self):
        """Test that from_string raises ValueError for invalid strings."""
        with pytest.raises(ValueError, match="Invalid time span string"):
            TimeSpan.from_string("invalid_span")

    def test_from_string_non_string(self):
        """Test that from_string raises TypeError for non-string inputs."""
        with pytest.raises(TypeError, match="Expected string"):
            TimeSpan.from_string(123)

    def test_get_time_unit(self):
        """Test that get_time_unit returns the correct TimeUnit."""
        assert TimeSpan.SPAN_HOURLY.get_time_unit() == TimeUnit("hours", 1)
        assert TimeSpan.SPAN_DAILY.get_time_unit() == TimeUnit("days", 1)

    def test_to_minutes(self):
        """Test conversion of TimeSpan to minutes."""
        assert TimeSpan.SPAN_HOURLY.to_minutes() == 60
        assert TimeSpan.SPAN_DAILY.to_minutes() == 24 * 60
        assert TimeSpan.SPAN_SEASONAL.to_minutes() == 90 * 24 * 60  # 3 months

    def test_str_representation(self):
        """Test string representation of TimeSpan members."""
        assert str(TimeSpan.SPAN_HALF_HOURLY) == "half_hourly"
        assert str(TimeSpan.SPAN_HOURLY) == "hourly"
        assert str(TimeSpan.SPAN_DAILY) == "daily"

    def test_comparison(self):
        """Test comparison of TimeSpan members."""
        assert TimeSpan.SPAN_HALF_HOURLY < TimeSpan.SPAN_HOURLY
        assert TimeSpan.SPAN_HOURLY < TimeSpan.SPAN_DAILY
        assert TimeSpan.SPAN_DAILY < TimeSpan.SPAN_WEEKLY
        assert TimeSpan.SPAN_WEEKLY < TimeSpan.SPAN_MONTHLY
        assert TimeSpan.SPAN_MONTHLY < TimeSpan.SPAN_YEARLY

    def test_comparison_invalid_type(self):
        """Test that comparison with invalid types returns NotImplemented."""
        assert TimeSpan.SPAN_HOURLY.__lt__("invalid") == NotImplemented

    def test_all_spans_ordered(self):
        """Test that all spans are properly ordered by duration."""
        spans = [
            TimeSpan.SPAN_HALF_HOURLY,
            TimeSpan.SPAN_HOURLY,
            TimeSpan.SPAN_THREE_HOURLY,
            TimeSpan.SPAN_SIX_HOURLY,
            TimeSpan.SPAN_DAILY,
            TimeSpan.SPAN_WEEKLY,
            TimeSpan.SPAN_MONTHLY,
            TimeSpan.SPAN_SEASONAL,
            TimeSpan.SPAN_YEARLY,
        ]
        for i in range(len(spans) - 1):
            assert spans[i] < spans[i + 1]
