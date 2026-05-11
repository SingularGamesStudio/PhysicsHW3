from dataclasses import dataclass
import time


@dataclass(slots=True)
class StepStats:
    dt_seconds: float = 0.0
    tps: float = 0.0
    sim_hz: float = 0.0
    contact_count: int = 0
    manifold_count: int = 0
    candidate_pair_count: int = 0


class SolveRateTracker:
    """
    Smoothed wall-clock physics throughput.
    This is actual solver steps per second, not 1/dt.
    """

    def __init__(self, smoothing=0.15):
        self.smoothing = float(smoothing)
        self.last_dt = 0.0
        self.tps = 0.0

    def begin(self):
        return time.perf_counter()

    def push_dt(self, dt_wall):
        dt_wall = max(1.0e-9, float(dt_wall))
        inst_tps = 1.0 / dt_wall
        if self.tps <= 0.0:
            self.tps = inst_tps
        else:
            s = self.smoothing
            self.tps = (1.0 - s) * self.tps + s * inst_tps
        self.last_dt = dt_wall
        return self.tps
