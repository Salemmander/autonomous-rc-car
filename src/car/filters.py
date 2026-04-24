"""
Signal filters for smoothing noisy predictions.
"""


class ScalarKalmanFilter1D:
    """
    1D Kalman filter for a scalar signal using a random-walk model.

    State:  x  (scalar estimate of the true value)
    Cov:    P  (scalar variance of the estimate)
    Q:      process noise   -- how much the true value can drift per step
    R:      measurement noise -- how noisy the incoming signal is

    Used here to smooth PilotNet's steering output before it reaches the servo.
    The ratio Q/R controls responsiveness: high Q/R = trust measurements (fast,
    jittery); low Q/R = trust the model's own estimate (smooth, laggy).
    """

    def __init__(
        self, x0: float = 0.0, P0: float = 1.0, Q: float = 0.02, R: float = 0.05
    ):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def step(self, measurement: float) -> float:
        """Run one predict + update cycle and return the filtered estimate."""
        x_pred = self.x
        P_pred = self.P + self.Q

        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1.0 - K) * P_pred
        return self.x

    def reset(self, x0: float = 0.0, P0: float = 1.0) -> None:
        """Reset state and covariance. Tuning params Q, R are preserved."""
        self.x = x0
        self.P = P0
