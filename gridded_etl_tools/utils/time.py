from dataclasses import dataclass
from typing import Literal, Dict, ClassVar, TypeVar
import re
from datetime import timedelta

TimeUnitType = Literal["minutes", "hours", "days", "weeks", "months", "years", "seasons"]


@dataclass(frozen=True)
class TimeUnit:
    """Represents a time unit with a specific value.

    Parameters
    ----------
    unit : TimeUnitType
        The time unit (minutes, hours, days, weeks, months, years, or seasons)
    value : int
        The number of units (must be positive)

    Raises
    ------
    ValueError
        If the value is not positive
    ValueError
        If the unit is not one of the valid time units
    """

    unit: TimeUnitType
    value: int

    # Conversion factors for different time units to minutes
    _CONVERSION_FACTORS: ClassVar[Dict[TimeUnitType, int]] = {
        "minutes": 1,
        "hours": 60,
        "days": 24 * 60,
        "weeks": 7 * 24 * 60,
        "months": 30 * 24 * 60,  # Approximation
        "years": 365 * 24 * 60,  # Approximation
        "seasons": 90 * 24 * 60,  # Approximation (3 months)
    }

    def __post_init__(self) -> None:
        """Validate the time unit value and unit type.

        Raises
        ------
        ValueError
            If the value is not positive
        ValueError
            If the unit is not one of the valid time units
        """
        if self.value <= 0:
            raise ValueError(f"Time unit value must be positive, got {self.value}")
        if self.unit not in self._CONVERSION_FACTORS:
            raise ValueError(f"Invalid time unit: {self.unit}. Must be one of {list(self._CONVERSION_FACTORS.keys())}")

    def to_minutes(self) -> int:
        """Convert this time unit to minutes.

        Returns
        -------
        int
            The number of minutes equivalent to this time unit

        Notes
        -----
        Conversions for months, years, and seasons are approximations:
        - months: 30 days
        - years: 365 days
        - seasons: 90 days (3 months)
        """
        if self.unit in ["months", "years", "seasons"]:
            raise ValueError(
                f"Cannot convert {self.unit} to minutes as {self.unit} is not of a fixed duration due to "
                "variable month lengths, leap years, etc. "
                "Please type manually in your code the specific duration you need"
            )
        return self.value * self._CONVERSION_FACTORS[self.unit]

    def __str__(self) -> str:
        """Return a string representation of this time unit.

        Returns
        -------
        str
            A string in the format "{value} {unit}"
        """
        return f"{self.value} {self.unit}"


T = TypeVar("T", bound="TimeSpan")


class TimeSpan:
    """Represents a time span with support for both predefined and arbitrary durations."""

    # Mapping of predefined TimeUnits to their string representations
    _PREDEFINED_STRINGS = {
        TimeUnit("minutes", 30): "half_hourly",
        TimeUnit("hours", 1): "hourly",
        TimeUnit("hours", 3): "3hourly",  # Keep for legacy support
        TimeUnit("hours", 6): "6hourly",  # Keep for legacy support
        TimeUnit("days", 1): "daily",
        TimeUnit("weeks", 1): "weekly",
        TimeUnit("months", 1): "monthly",
        TimeUnit("years", 1): "yearly",
        TimeUnit("seasons", 1): "seasonal",
    }

    def __init__(self, time_unit: TimeUnit):
        self.time_unit = time_unit

    @classmethod
    def create(cls, unit: TimeUnitType, value: int) -> "TimeSpan":
        """Create a TimeSpan for arbitrary duration.

        Parameters
        ----------
        unit : TimeUnitType
            The time unit (minutes, hours, days, weeks, months, years, or seasons)
        value : int
            The number of units (must be positive)

        Returns
        -------
        TimeSpan
            A TimeSpan representing the specified duration
        """
        return cls(TimeUnit(unit, value))

    @classmethod
    def from_string(cls, span_str: str) -> "TimeSpan":
        """Convert a string representation to a TimeSpan.

        Supports both predefined spans and arbitrary durations.

        Parameters
        ----------
        span_str : str
            String representation (e.g., "2minutes", "hourly", "15minutes")

        Returns
        -------
        TimeSpan
            The corresponding TimeSpan
        """
        if not isinstance(span_str, str):
            raise TypeError(f"Expected string, got {type(span_str).__name__}")

        # First try predefined spans
        predefined_map = {
            "half_hourly": cls.create("minutes", 30),
            "hourly": cls.create("hours", 1),
            "3hourly": cls.create("hours", 3),
            "6hourly": cls.create("hours", 6),
            "daily": cls.create("days", 1),
            "weekly": cls.create("weeks", 1),
            "monthly": cls.create("months", 1),
            "yearly": cls.create("years", 1),
            "seasonal": cls.create("seasons", 1),
        }

        if span_str in predefined_map:
            return predefined_map[span_str]

        # Try to parse as arbitrary duration (e.g., "15minutes", "2hours", "2minutes")
        pattern = r"^(\d+)(minutes?|hours?|days?|weeks?|months?|years?|seasons?)$"
        match = re.match(pattern, span_str.lower())

        if match:
            value = int(match.group(1))
            unit_with_s = match.group(2)
            # Ensure unit ends with 's' for consistency
            unit = unit_with_s if unit_with_s.endswith("s") else unit_with_s + "s"
            return cls.create(unit, value)

        valid_spans = ", ".join(sorted(predefined_map.keys()))
        raise ValueError(
            f"Invalid time span string: '{span_str}'. Must be one of: {valid_spans} or a pattern like '15minutes'"
        )

    def get_time_unit(self) -> TimeUnit:
        """Get the TimeUnit associated with this TimeSpan."""
        return self.time_unit

    def to_minutes(self) -> int:
        """Convert this time span to minutes."""
        return self.time_unit.to_minutes()

    def __str__(self) -> str:
        """Return a string representation of this time span."""
        # Check if this is a predefined span
        if self.time_unit in self._PREDEFINED_STRINGS:
            return self._PREDEFINED_STRINGS[self.time_unit]

        # For arbitrary spans, return a generic format
        return f"{self.time_unit.value}{self.time_unit.unit}"

    def __lt__(self, other: "TimeSpan") -> bool:
        """Compare time spans based on their duration in minutes."""
        if not isinstance(other, TimeSpan):
            return NotImplemented
        return self.to_minutes() < other.to_minutes()

    def to_timedelta(self) -> timedelta:
        """Convert this time span to a timedelta."""
        return timedelta(minutes=self.to_minutes())

    def __eq__(self, other: object) -> bool:
        """Compare TimeSpan objects for equality."""
        if not isinstance(other, TimeSpan):
            return NotImplemented
        return self.time_unit == other.time_unit


# Create class attributes for backward compatibility
TimeSpan.SPAN_HALF_HOURLY = TimeSpan.create("minutes", 30)
TimeSpan.SPAN_HOURLY = TimeSpan.create("hours", 1)
TimeSpan.SPAN_THREE_HOURLY = TimeSpan.create("hours", 3)
TimeSpan.SPAN_SIX_HOURLY = TimeSpan.create("hours", 6)
TimeSpan.SPAN_DAILY = TimeSpan.create("days", 1)
TimeSpan.SPAN_WEEKLY = TimeSpan.create("weeks", 1)
TimeSpan.SPAN_MONTHLY = TimeSpan.create("months", 1)
TimeSpan.SPAN_YEARLY = TimeSpan.create("years", 1)
TimeSpan.SPAN_SEASONAL = TimeSpan.create("seasons", 1)
