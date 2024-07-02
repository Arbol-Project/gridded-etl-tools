class NanFrequencyMismatchError(Exception):
    """Exception raised when the frequency of NaNs does not match the expected distribution."""

    def __init__(self, observed_frequency, expected_frequency, lower_bound, upper_bound):
        self.observed_frequency = observed_frequency
        self.expected_frequency = expected_frequency
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.message = (
            f"Observed NaN frequency {observed_frequency:.4f} with 99.9% CI "
            f"[{lower_bound:.4f}, {upper_bound:.4f}] does not match "
            f"the expected frequency {expected_frequency:.4f}."
        )
        super().__init__(self.message)
