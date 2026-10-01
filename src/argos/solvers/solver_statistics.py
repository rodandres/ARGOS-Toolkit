from dataclasses import dataclass, field
import numpy as np


@dataclass
class SolverStatistics:
    nfev: list[int] = field(default_factory=list)
    accepted_steps: list[int] = field(default_factory=list)
    rejected_steps: list[int] = field(default_factory=list)
    elapsed_time: list[float] = field(default_factory=list)

    def add(
        self,
        nfev: int,
        accepted_steps: int,
        rejected_steps: int,
        elapsed_time: float,
    ):
        self.nfev.append(nfev)
        self.accepted_steps.append(accepted_steps)
        self.rejected_steps.append(rejected_steps)
        self.elapsed_time.append(elapsed_time)

    def summary(self):
        data = {
            "nfev": np.asarray(self.nfev),
            "accepted_steps": np.asarray(self.accepted_steps),
            "rejected_steps": np.asarray(self.rejected_steps),
            "elapsed_time": np.asarray(self.elapsed_time),
        }

        summary = {}

        for name, values in data.items():
            summary[name] = {
                "total": np.sum(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "p25": np.percentile(values, 25),
                "p75": np.percentile(values, 75),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
                "max": np.max(values),
            }

        return summary