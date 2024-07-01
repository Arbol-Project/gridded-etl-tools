class NanFrequencyMismatchError(Exception):
    """Exception raised when the frequency of NaNs does not match the expected distribution."""

    def __init__(self, observed_frequency, expected_frequency, p_value):
        self.observed_frequency = observed_frequency
        self.expected_frequency = expected_frequency
        self.p_value = p_value
        self.message = (
            f"Observed NaN frequency {observed_frequency:.2f} does not match "
            f"the expected frequency {expected_frequency:.2f}. "
            f"P-value: {p_value:.4f}"
        )
        super().__init__(self.message)
