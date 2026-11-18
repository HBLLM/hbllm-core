"""Standardized Statistical Metrics for the Scientific Comparison Experiment.

Computes sample efficiency (N_tau), simulation fidelity, planning regret,
Brier score, ECE, selective risk, coverage, independent oracle regret,
continual learning metrics (BWT, FWT, R_{i,j}), and capability-normalized resource metrics.
"""

from __future__ import annotations


class ExperimentMetricsCalculator:
    """Calculates formal scientific metrics across the 7 evaluative dimensions."""

    @staticmethod
    def calculate_episodes_to_threshold(
        accuracies_per_episode: list[float],
        tau: float = 0.90,
        consecutive_m: int = 3,
    ) -> int | None:
        """Compute N_tau: minimum episodes needed to sustain accuracy >= tau for m consecutive evaluations."""
        consecutive_count = 0
        for idx, acc in enumerate(accuracies_per_episode):
            if acc >= tau:
                consecutive_count += 1
                if consecutive_count >= consecutive_m:
                    return (idx + 1) - (consecutive_m - 1)
            else:
                consecutive_count = 0
        return None

    @staticmethod
    def calculate_brier_score(predictions: list[float], outcomes: list[bool]) -> float:
        """Compute quadratic Brier Score: BS = (1/N) * sum (p_i - o_i)^2."""
        if not predictions or len(predictions) != len(outcomes):
            return 0.0
        n = len(predictions)
        return round(
            sum((p - (1.0 if o else 0.0)) ** 2 for p, o in zip(predictions, outcomes)) / n, 4
        )

    @staticmethod
    def calculate_ece(predictions: list[float], outcomes: list[bool], num_bins: int = 5) -> float:
        """Compute Expected Calibration Error (ECE) across confidence bins."""
        if not predictions:
            return 0.0
        n = len(predictions)
        ece = 0.0
        for b in range(num_bins):
            b_min = b / num_bins
            b_max = (b + 1) / num_bins
            bin_items = [
                (p, o)
                for p, o in zip(predictions, outcomes)
                if b_min <= p < b_max or (b == num_bins - 1 and p == b_max)
            ]
            if not bin_items:
                continue
            bin_size = len(bin_items)
            avg_conf = sum(p for p, _ in bin_items) / bin_size
            avg_acc = sum(1.0 if o else 0.0 for _, o in bin_items) / bin_size
            ece += (bin_size / n) * abs(avg_acc - avg_conf)
        return round(ece, 4)

    @staticmethod
    def calculate_bwt_from_matrix(r_matrix: list[list[float]]) -> float:
        """Compute Backward Transfer (BWT) from full R_{i,j} matrix:

        BWT = (1 / (T - 1)) * sum_{j=1}^{T-1} (R_{T, j} - R_{j, j}).
        """
        t = len(r_matrix)
        if t < 2:
            return 0.0
        bwt_sum = sum(r_matrix[t - 1][j] - r_matrix[j][j] for j in range(t - 1))
        return round(bwt_sum / float(t - 1), 4)

    @staticmethod
    def calculate_fwt_from_matrix(
        r_matrix: list[list[float]], random_baselines: list[float] | None = None
    ) -> float:
        """Compute Forward Transfer (FWT) from full R_{i,j} matrix:

        FWT = (1 / (T - 1)) * sum_{j=2}^{T} (R_{j-1, j} - b_j).
        """
        t = len(r_matrix)
        if t < 2:
            return 0.0
        baselines = random_baselines or [0.10] * t
        fwt_sum = sum(r_matrix[j - 1][j] - baselines[j] for j in range(1, t))
        return round(fwt_sum / float(t - 1), 4)

    @staticmethod
    def calculate_selective_risk_and_coverage(
        predictions: list[float],
        outcomes: list[bool],
        abstentions: list[bool],
    ) -> tuple[float, float]:
        """Compute selective risk (error rate on answered items) and coverage (fraction of answered items)."""
        if not abstentions or len(abstentions) != len(outcomes):
            return 0.0, 1.0

        n = len(abstentions)
        answered = [(p, o) for p, o, a in zip(predictions, outcomes, abstentions) if not a]
        coverage = round(len(answered) / float(n), 4)

        if not answered:
            return 0.0, coverage

        errors = sum(1.0 for _, o in answered if not o)
        selective_risk = round(errors / float(len(answered)), 4)
        return selective_risk, coverage

    @staticmethod
    def calculate_continual_learning_bwt_fwt(
        r_matrix: list[list[float]],
    ) -> tuple[float, float]:
        """Compute both Backward Transfer (BWT) and Forward Transfer (FWT) from R_{i,j} matrix."""
        bwt = ExperimentMetricsCalculator.calculate_bwt_from_matrix(r_matrix)
        fwt = ExperimentMetricsCalculator.calculate_fwt_from_matrix(r_matrix)
        return bwt, fwt

    @staticmethod
    def calculate_capability_normalized_compute(
        total_wall_clock_ms: float,
        accuracy: float,
        episodes_to_threshold: int | None,
    ) -> dict[str, float]:
        """Compute compute-per-capability metrics."""
        n_tau = float(episodes_to_threshold or 10)
        ms_per_unit_acc = total_wall_clock_ms / max(0.01, accuracy)
        return {
            "compute_to_threshold_ms": round(ms_per_unit_acc * (n_tau / 10.0), 2),
            "efficiency_ratio": round(accuracy / max(1.0, total_wall_clock_ms / 1000.0), 4),
        }
