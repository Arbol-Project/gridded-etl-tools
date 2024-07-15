class NanFrequencyMismatchError(Exception):
    """Exception raised when the frequency of NaNs does not match the expected distribution."""

    def __init__(self, observed_frequency, expected_frequency, lower_bound, upper_bound, alpha, time_value):
        confidence_interval = (1 - alpha) * 100
        self.message = (
            f"Observed NaN frequency {observed_frequency:.4f} with {confidence_interval:.4f}% CI "
            f"[{lower_bound:.4f}, {upper_bound:.4f}] for time step {time_value.astype(str).split('.')[0]} does not match "
            f"the expected frequency {expected_frequency:.4f}."
        )
        super().__init__(self.message)
