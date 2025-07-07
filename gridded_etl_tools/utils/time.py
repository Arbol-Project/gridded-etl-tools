from dataclasses import dataclass
from typing import Literal, Dict, ClassVar, TypeVar
from enum import Enum
from datetime import timedelta

TimeUnitType = Literal["minutes", "hours", "days", "weeks", "months", "years", "seasons"]


@dataclass
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


class TimeSpan(Enum):
    """Enumeration of common time spans used in climate data.

    Each span represents a specific time interval that can be used to group or aggregate data.
    The time spans are ordered from smallest to largest interval.
    """

    SPAN_TWO_MINUTES = TimeUnit("minutes", 2)
    SPAN_HALF_HOURLY = TimeUnit("minutes", 30)
    SPAN_HOURLY = TimeUnit("hours", 1)
    SPAN_THREE_HOURLY = TimeUnit("hours", 3)
    SPAN_SIX_HOURLY = TimeUnit("hours", 6)
    SPAN_DAILY = TimeUnit("days", 1)
    SPAN_WEEKLY = TimeUnit("weeks", 1)
    SPAN_MONTHLY = TimeUnit("months", 1)
    SPAN_YEARLY = TimeUnit("years", 1)
    SPAN_SEASONAL = TimeUnit("seasons", 1)  # 3 months

    @classmethod
    def from_string(cls: type[T], span_str: str) -> T:
        """Convert a string representation of a time span to the corresponding TimeSpan enum.

        This is useful for converting user-provided time spans to the internal representation
        and hence maintaining backwards compatibility with existing code.

        Parameters
        ----------
        span_str : str
            String representation of the time span (e.g., "hourly", "daily")

        Returns
        -------
        TimeSpan
            The corresponding TimeSpan enum member

        Raises
        ------
        TypeError
            If span_str is not a string
        ValueError
            If span_str does not correspond to a valid time span
        """
        if not isinstance(span_str, str):
            raise TypeError(f"Expected string, got {type(span_str).__name__}")

        # Create mapping of string representations to enum members on the fly to avoid class variable issues
        string_map = {
            "2minutes": cls.SPAN_TWO_MINUTES,
            "half_hourly": cls.SPAN_HALF_HOURLY,
            "hourly": cls.SPAN_HOURLY,
            "3hourly": cls.SPAN_THREE_HOURLY,
            "6hourly": cls.SPAN_SIX_HOURLY,
            "daily": cls.SPAN_DAILY,
            "weekly": cls.SPAN_WEEKLY,
            "monthly": cls.SPAN_MONTHLY,
            "yearly": cls.SPAN_YEARLY,
            "seasonal": cls.SPAN_SEASONAL,
        }
        if span_str not in string_map:
            valid_spans = ", ".join(sorted(string_map.keys()))
            raise ValueError(f"Invalid time span string: '{span_str}'. " f"Must be one of: {valid_spans}")
        return string_map[span_str]

    def get_time_unit(self) -> TimeUnit:
        """Get the TimeUnit associated with this TimeSpan.

        Returns
        -------
        TimeUnit
            The time unit representing this span
        """
        return self.value

    def to_minutes(self) -> int:
        """Convert this time span to minutes.

        Returns
        -------
        int
            The number of minutes in this time span

        Notes
        -----
        Conversions for months, years, and seasons are approximations:
        - months: 30 days
        - years: 365 days
        - seasons: 90 days (3 months)
        """
        return self.value.to_minutes()

    def __str__(self) -> str:
        """Return a string representation of this time span.

        Returns
        -------
        str
            A string representation of the time span without the "SPAN_" prefix
            and with numeric values converted from words (e.g., "THREE_" becomes "3")
        """
        # First remove the SPAN_ prefix and convert to lowercase
        name = self.name.lower().replace("span_", "")

        # Dictionary mapping word numbers to digits
        # NOTE there are packages that do this programmatically,
        # but installing a package for this is overkill
        word_to_num = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }

        # Replace any word numbers with their digit equivalents
        for word, num in word_to_num.items():
            if name.startswith(word + "_"):
                name = name.replace(word + "_", num)

        return name

    def __lt__(self, other: "TimeSpan") -> bool:
        """Compare time spans based on their duration in minutes.

        Parameters
        ----------
        other : TimeSpan
            Another TimeSpan to compare with

        Returns
        -------
        bool
            True if this time span is shorter than the other

        Raises
        ------
        TypeError
            If other is not a TimeSpan
        """
        if not isinstance(other, TimeSpan):
            return NotImplemented
        return self.to_minutes() < other.to_minutes()

    def to_timedelta(self) -> timedelta:
        """Convert this time span to a timedelta.

        Returns
        -------
        timedelta
            The timedelta representing this time span
        """
        return timedelta(minutes=self.to_minutes())
