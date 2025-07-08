import pytest
from datetime import timedelta
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
            ("minutes", 30, 30),
            ("hours", 2, 120),
            ("days", 2, 48 * 60),
        ],
    )
    def test_to_minutes(self, unit, value, expected_minutes):
        """Test conversion of various time units to minutes."""
        time_unit = TimeUnit(unit, value)
        assert time_unit.to_minutes() == expected_minutes

    @pytest.mark.parametrize(
        "unit,value,expected_minutes",
        [
            ("months", 1, 30 * 24 * 60),  # Approximation
            ("years", 1, 365 * 24 * 60),  # Approximation
            ("seasons", 1, 90 * 24 * 60),  # Approximation
        ],
    )
    def test_to_minutes_invalid(self, unit, value, expected_minutes):
        """Test conversion of various time units to minutes."""
        time_unit = TimeUnit(unit, value)
        with pytest.raises(
            ValueError,
            match=f"Cannot convert {time_unit.unit} to minutes as {time_unit.unit} is not of a fixed duration.",
        ):
            time_unit.to_minutes()

    def test_str_representation(self):
        """Test string representation of TimeUnit."""
        unit = TimeUnit("hours", 2)
        assert str(unit) == "2 hours"


class TestTimeSpan:
    """Test suite for TimeSpan enum."""

    def test_enum_members(self):
        """Test that all expected TimeSpan members exist with correct values."""
        assert TimeSpan.SPAN_HALF_HOURLY.time_unit == TimeUnit("minutes", 30)
        assert TimeSpan.SPAN_HOURLY.time_unit == TimeUnit("hours", 1)
        assert TimeSpan.SPAN_THREE_HOURLY.time_unit == TimeUnit("hours", 3)
        assert TimeSpan.SPAN_SIX_HOURLY.time_unit == TimeUnit("hours", 6)
        assert TimeSpan.SPAN_DAILY.time_unit == TimeUnit("days", 1)
        assert TimeSpan.SPAN_WEEKLY.time_unit == TimeUnit("weeks", 1)
        assert TimeSpan.SPAN_MONTHLY.time_unit == TimeUnit("months", 1)
        assert TimeSpan.SPAN_YEARLY.time_unit == TimeUnit("years", 1)
        assert TimeSpan.SPAN_SEASONAL.time_unit == TimeUnit("seasons", 1)

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
        with pytest.raises(
            ValueError, match="Cannot convert seasons to minutes as seasons is not of a fixed duration."
        ):
            TimeSpan.SPAN_SEASONAL.to_minutes()

    def test_str_representation(self):
        """Test string representation of TimeSpan members."""
        assert str(TimeSpan.SPAN_HALF_HOURLY) == "half_hourly"
        assert str(TimeSpan.SPAN_HOURLY) == "hourly"
        assert str(TimeSpan.SPAN_THREE_HOURLY) == "3hourly"
        assert str(TimeSpan.SPAN_SIX_HOURLY) == "6hourly"
        assert str(TimeSpan.SPAN_DAILY) == "daily"
        assert str(TimeSpan.SPAN_WEEKLY) == "weekly"
        assert str(TimeSpan.SPAN_MONTHLY) == "monthly"
        assert str(TimeSpan.SPAN_YEARLY) == "yearly"
        assert str(TimeSpan.SPAN_SEASONAL) == "seasonal"

    def test_comparison(self):
        """Test comparison of TimeSpan members."""
        assert TimeSpan.SPAN_HALF_HOURLY < TimeSpan.SPAN_HOURLY
        assert TimeSpan.SPAN_HOURLY < TimeSpan.SPAN_DAILY
        assert TimeSpan.SPAN_DAILY < TimeSpan.SPAN_WEEKLY
        with pytest.raises(ValueError, match="Cannot convert months to minutes as months is not of a fixed duration."):
            TimeSpan.SPAN_WEEKLY < TimeSpan.SPAN_MONTHLY
        with pytest.raises(ValueError, match="Cannot convert months to minutes as months is not of a fixed duration."):
            TimeSpan.SPAN_MONTHLY < TimeSpan.SPAN_YEARLY
        with pytest.raises(
            ValueError, match="Cannot convert seasons to minutes as seasons is not of a fixed duration."
        ):
            TimeSpan.SPAN_SEASONAL < TimeSpan.SPAN_YEARLY

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
            # MONTHLY, YEARLY, SEASONAL cannot be compared mathematically due to variable durations
        ]
        for i in range(len(spans) - 1):
            assert spans[i] < spans[i + 1]

    def test_to_timedelta(self):
        """Test conversion of TimeSpan to timedelta."""
        assert TimeSpan.SPAN_HALF_HOURLY.to_timedelta() == timedelta(minutes=30)
        assert TimeSpan.SPAN_HOURLY.to_timedelta() == timedelta(hours=1)
        assert TimeSpan.SPAN_THREE_HOURLY.to_timedelta() == timedelta(hours=3)
        assert TimeSpan.SPAN_SIX_HOURLY.to_timedelta() == timedelta(hours=6)
        assert TimeSpan.SPAN_DAILY.to_timedelta() == timedelta(days=1)
        assert TimeSpan.SPAN_WEEKLY.to_timedelta() == timedelta(weeks=1)
        with pytest.raises(ValueError, match="Cannot convert months to minutes as months is not of a fixed duration"):
            TimeSpan.SPAN_MONTHLY.to_timedelta()
        with pytest.raises(ValueError, match="Cannot convert years to minutes as years is not of a fixed duration"):
            TimeSpan.SPAN_YEARLY.to_timedelta()
        with pytest.raises(
            ValueError, match="Cannot convert seasons to minutes as seasons is not of a fixed duration"
        ):
            TimeSpan.SPAN_SEASONAL.to_timedelta()

    def test_create_factory_method(self):
        """Test the create factory method for arbitrary time spans."""
        # Test various time units
        two_minutes = TimeSpan.create("minutes", 2)
        assert two_minutes.time_unit == TimeUnit("minutes", 2)
        assert two_minutes.to_minutes() == 2
        assert str(two_minutes) == "2minutes"

        fifteen_minutes = TimeSpan.create("minutes", 15)
        assert fifteen_minutes.time_unit == TimeUnit("minutes", 15)
        assert fifteen_minutes.to_minutes() == 15
        assert str(fifteen_minutes) == "15minutes"

        three_hours = TimeSpan.create("hours", 3)
        assert three_hours.time_unit == TimeUnit("hours", 3)
        assert three_hours.to_minutes() == 180
        assert str(three_hours) == "3hourly"  # Legacy format

        two_days = TimeSpan.create("days", 2)
        assert two_days.time_unit == TimeUnit("days", 2)
        assert two_days.to_minutes() == 2880  # 2 * 24 * 60
        assert str(two_days) == "2days"

        one_week = TimeSpan.create("weeks", 1)
        assert one_week.time_unit == TimeUnit("weeks", 1)
        assert one_week.to_minutes() == 10080  # 7 * 24 * 60
        assert str(one_week) == "weekly"  # Predefined format

        # Test edge cases
        one_minute = TimeSpan.create("minutes", 1)
        assert one_minute.time_unit == TimeUnit("minutes", 1)
        assert one_minute.to_minutes() == 1
        assert str(one_minute) == "1minutes"

        large_hours = TimeSpan.create("hours", 100)
        assert large_hours.time_unit == TimeUnit("hours", 100)
        assert large_hours.to_minutes() == 6000
        assert str(large_hours) == "100hours"

    def test_create_factory_method_invalid_inputs(self):
        """Test that create factory method properly validates inputs."""
        # Test invalid units
        with pytest.raises(ValueError, match="Invalid time unit: invalid_unit"):
            TimeSpan.create("invalid_unit", 1)

        # Test negative values
        with pytest.raises(ValueError, match="Time unit value must be positive"):
            TimeSpan.create("minutes", -1)

        # Test zero values
        with pytest.raises(ValueError, match="Time unit value must be positive"):
            TimeSpan.create("hours", 0)

        # Test non-integer values (should raise TypeError)
        with pytest.raises(TypeError):
            TimeSpan.create("minutes", "2")  # type: ignore

    def test_create_factory_method_comparison(self):
        """Test that created TimeSpan objects can be compared."""
        two_minutes = TimeSpan.create("minutes", 2)
        thirty_minutes = TimeSpan.create("minutes", 30)
        one_hour = TimeSpan.create("hours", 1)

        # Test comparisons
        assert two_minutes < thirty_minutes
        assert thirty_minutes < one_hour
        assert two_minutes < one_hour

        # Test equality
        assert two_minutes == TimeSpan.create("minutes", 2)
        assert two_minutes != TimeSpan.create("minutes", 3)

    def test_create_factory_method_timedelta(self):
        """Test that created TimeSpan objects can be converted to timedelta."""
        two_minutes = TimeSpan.create("minutes", 2)
        three_hours = TimeSpan.create("hours", 3)
        one_day = TimeSpan.create("days", 1)

        assert two_minutes.to_timedelta() == timedelta(minutes=2)
        assert three_hours.to_timedelta() == timedelta(hours=3)
        assert one_day.to_timedelta() == timedelta(days=1)

    def test_create_factory_method_with_variable_duration_units(self):
        """Test create factory method with units that can't be converted to minutes."""
        # These should raise ValueError when trying to convert to minutes
        monthly = TimeSpan.create("months", 1)
        yearly = TimeSpan.create("years", 1)
        seasonal = TimeSpan.create("seasons", 1)

        # They should be created successfully
        assert monthly.time_unit == TimeUnit("months", 1)
        assert yearly.time_unit == TimeUnit("years", 1)
        assert seasonal.time_unit == TimeUnit("seasons", 1)

        # But should raise ValueError when converting to minutes
        with pytest.raises(ValueError, match="Cannot convert months to minutes"):
            monthly.to_minutes()

        with pytest.raises(ValueError, match="Cannot convert years to minutes"):
            yearly.to_minutes()

        with pytest.raises(ValueError, match="Cannot convert seasons to minutes"):
            seasonal.to_minutes()

    def test_create_factory_method_string_representation(self):
        """Test that created TimeSpan objects have correct string representations."""
        # Test arbitrary spans (not in predefined list)
        two_minutes = TimeSpan.create("minutes", 2)
        fifteen_minutes = TimeSpan.create("minutes", 15)
        # Note: 3 hours is predefined, so it uses the legacy format
        three_hours = TimeSpan.create("hours", 3)

        assert str(two_minutes) == "2minutes"
        assert str(fifteen_minutes) == "15minutes"
        assert str(three_hours) == "3hourly"  # Legacy format

        # Test predefined spans (should use predefined string)
        thirty_minutes = TimeSpan.create("minutes", 30)
        one_hour = TimeSpan.create("hours", 1)
        one_day = TimeSpan.create("days", 1)
        one_week = TimeSpan.create("weeks", 1)

        assert str(thirty_minutes) == "half_hourly"
        assert str(one_hour) == "hourly"
        assert str(one_day) == "daily"
        assert str(one_week) == "weekly"  # Predefined format

    def test_create_factory_method_from_string_integration(self):
        """Test integration between create factory method and from_string."""
        # Test that from_string can parse arbitrary durations
        two_minutes_from_string = TimeSpan.from_string("2minutes")
        two_minutes_from_create = TimeSpan.create("minutes", 2)
        assert two_minutes_from_string == two_minutes_from_create

        fifteen_minutes_from_string = TimeSpan.from_string("15minutes")
        fifteen_minutes_from_create = TimeSpan.create("minutes", 15)
        assert fifteen_minutes_from_string == fifteen_minutes_from_create

        # Test that 3hours uses the legacy format
        three_hours_from_string = TimeSpan.from_string("3hourly")  # Use legacy format
        three_hours_from_create = TimeSpan.create("hours", 3)
        assert three_hours_from_string == three_hours_from_create

    def test_equality_with_non_timespan(self):
        """Test that equality comparison with non-TimeSpan objects returns NotImplemented."""
        timespan = TimeSpan.create("minutes", 2)

        # Test with various non-TimeSpan objects
        assert timespan.__eq__("not a timespan") == NotImplemented
        assert timespan.__eq__(123) == NotImplemented
        assert timespan.__eq__(None) == NotImplemented
        assert timespan.__eq__(TimeUnit("minutes", 2)) == NotImplemented

        # Test the == operator (should use __eq__)
        assert not (timespan == "not a timespan")
        assert not (timespan == 123)
        assert not (timespan is None)
        assert not (timespan == TimeUnit("minutes", 2))
