# SPDX-License-Identifier: Apache-2.0
"""Causal, host-side planning for a small set of pre-captured URMA paths.

This module never reads a device tensor or changes a captured graph. The caller
supplies already-completed feedback and only paths with compatible persistent
Index/cache state. In particular, standalone serial/fused replay correctness
does not establish that serving can alternate their layouts without conversion.
"""

from dataclasses import dataclass
import math


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class Demand:
    """Prediction in unique token reads, not Q4 TopK entries or a miss fraction."""

    expected: float
    deviation: float
    samples: int


class DemandHistory:
    """Track a request lifetime and layer group without using this step's oracle."""

    def __init__(self, alpha=0.5, max_entries=4096):
        if not 0 < alpha <= 1 or type(max_entries) is not int or max_entries < 1:
            raise ValueError("Require 0 < alpha <= 1 and a positive history bound")
        self.alpha, self.max_entries = alpha, max_entries
        self._rows = {}

    def observe(self, identity, group, step, *, selected_reads):
        """identity must include request generation; selected_reads excludes waste.

        A service exposing only prefetch+final traffic must not pass it here as
        selected_reads. Obtain a valid selected-demand counter or retain the
        calibrated fallback; do not insert a synchronous device readback.
        """
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        value = _nonnegative(selected_reads, "selected_reads")
        key = (identity, group)
        old = self._rows.get(key)
        if old and step <= old[0]:
            raise ValueError("Feedback must advance monotonically within a lifetime")
        if old and step == old[0] + 1:
            _, mean, deviation, count = old
            error = abs(value - mean)
            mean += self.alpha * (value - mean)
            deviation += self.alpha * (error - deviation)
            count += 1
        else:
            mean, deviation, count = value, 0.0, 1
        self._rows.pop(key, None)
        self._rows[key] = (step, mean, deviation, count)
        while len(self._rows) > self.max_entries:
            self._rows.pop(next(iter(self._rows)))

    def predict(self, identity, group, step, *, fallback_reads):
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        fallback = _nonnegative(fallback_reads, "fallback_reads")
        old = self._rows.get((identity, group))
        if old and step <= old[0]:
            raise ValueError("Prediction must precede feedback for the target step")
        if old and step == old[0] + 1:
            return Demand(old[1], old[2], old[3])
        return Demand(fallback, fallback * 0.5, 0)


def allocate_controls(predictions, *, pool_tokens, window_limit=128,
                      key_steps=0, carry=True, uncertainty_weight=1.0):
    """Distribute a bounded pool across owners before capture/replay submission.

    Each owner has its own 2048-slot ceiling; unused quota can be borrowed from
    other owners at planning time. Hardware still enforces slot/transport limits.
    This is not an atomic device-wide pool and cannot steal a slot already used
    by another request. Allocation is O(B log B), with B <= 16 currently.
    """
    if not predictions or len(predictions) > 16:
        raise ValueError("Current streaming implementation supports BS1..16")
    if type(pool_tokens) is not int or not 0 <= pool_tokens <= len(predictions) * 2048:
        raise ValueError("Pool must fit the current owner's 2048-slot capacities")
    if type(window_limit) is not int or not 1 <= window_limit <= 480:
        raise ValueError("window_limit must be in 1..480")
    if type(key_steps) is not int or not 0 <= key_steps <= 65535 or type(carry) is not bool:
        raise ValueError("Require integer ordered-key steps and a boolean carry policy")
    _nonnegative(uncertainty_weight, "uncertainty_weight")
    wishes = []
    for row in predictions:
        mean = _nonnegative(row.expected, "expected")
        deviation = _nonnegative(row.deviation, "deviation")
        wishes.append(min(2048, math.ceil(mean + uncertainty_weight * deviation)))
    total = sum(wishes)
    if total <= pool_tokens:
        allocations = wishes
    else:
        exact = [wanted * pool_tokens / total for wanted in wishes]
        allocations = [math.floor(value) for value in exact]
        remainder = pool_tokens - sum(allocations)
        order = sorted(range(len(wishes)), key=lambda i: (-(exact[i] - allocations[i]), i))
        for index in order[:remainder]:
            allocations[index] += 1
    return [(count, window_limit, key_steps, int(carry)) for count in allocations]


def estimate_cost(points, demand):
    """Interpolate measured FULL-chain costs inside a calibrated demand range.

    points are (selected read demand, median microseconds) for the same layer
    group or independently validated workload class, shape, kernel/runtime,
    workers and admission policy. Equal total reads do not imply equal LI or
    tail costs: do not fit different groups into one demand-only curve without
    validating that class. No BS/history/payload/NUMA extrapolation is performed.
    """
    demand = _nonnegative(demand, "demand")
    buckets = {}
    for count, latency in points:
        count = _nonnegative(count, "calibration demand")
        latency = _nonnegative(latency, "full-chain latency")
        if latency == 0:
            raise ValueError("Measured latency must be positive")
        buckets.setdefault(count, []).append(latency)
    ordered = sorted((count, sum(values) / len(values)) for count, values in buckets.items())
    if not ordered or demand < ordered[0][0] or demand > ordered[-1][0]:
        return None
    for count, latency in ordered:
        if demand == count:
            return latency
    for (left, a), (right, b) in zip(ordered, ordered[1:]):
        if left < demand < right:
            return a + (b - a) * (demand - left) / (right - left)
    return None


class PathSelector:
    """Choose among compatible captured paths with hysteresis and full costs.

    The caller owns the small graph bank and validates state transitions. A
    calibration fingerprint MUST identify the layer group or an independently
    validated workload class, as well as production/Get hashes, hardware/NUMA,
    BS, query packing, heads, payload width, history and admission policy. Five
    groups with similar read counts are not automatically one workload class.
    The caller makes one decision per decode step using completed feedback.
    Mean +/- deviation is a conservative decision band, not a statistical
    confidence guarantee. This class only chooses a name; it never claims
    offline reset/conversion is free in serving.
    """

    def __init__(self, *, fingerprint, curves, default="serial", margin=0.05, confirmations=2):
        if (default not in curves or not 0 <= margin < 1 or
                type(confirmations) is not int or confirmations < 1):
            raise ValueError("Invalid calibrated default or hysteresis")
        self.fingerprint, self.curves, self.default = fingerprint, curves, default
        self.margin, self.confirmations = margin, confirmations
        self.current, self.pending, self.count = default, None, 0

    def choose(self, *, fingerprint, predictions, compatible_paths, transition_us=0.0):
        transition_us = _nonnegative(transition_us, "transition_us")
        allowed = set(compatible_paths)
        if self.current not in allowed:
            raise ValueError("Current graph state has no validated compatible path")
        if fingerprint != self.fingerprint:
            self.pending, self.count = None, 0
            return dict(path=self.current, reason="uncalibrated_shape_or_runtime", predicted_us=None)
        predictions = tuple(predictions)
        if not predictions:
            raise ValueError("At least one request prediction is required")
        means, deviations = [], []
        for row in predictions:
            means.append(_nonnegative(row.expected, "expected"))
            deviations.append(_nonnegative(row.deviation, "deviation"))
            if type(row.samples) is not int or row.samples < 0:
                raise ValueError("samples must be a nonnegative integer")
        demand = sum(means)
        lower = sum(max(0.0, mean - deviation) for mean, deviation in zip(means, deviations))
        upper = sum(mean + deviation for mean, deviation in zip(means, deviations))
        probes = (lower, demand, upper)
        estimates = {
            name: tuple(estimate_cost(points, count) for count in probes)
            for name, points in self.curves.items() if name in allowed
        }
        costs = {name: values[1] for name, values in estimates.items()}
        costs = {name: cost for name, cost in costs.items() if cost is not None}
        if self.current not in costs:
            self.pending, self.count = None, 0
            return dict(path=self.current, reason="outside_calibrated_demand", predicted_us=None)
        original = self.current
        reason = "measured_full_cost_with_hysteresis"
        eligible = []
        if any(row.samples == 0 for row in predictions):
            reason = "cold_start"
        elif any(cost is None for cost in estimates[self.current]):
            reason = "outside_calibrated_uncertainty"
        else:
            eligible = [
                name for name, values in estimates.items() if name != self.current
                and all(candidate is not None and
                        candidate + transition_us < current * (1 - self.margin)
                        for candidate, current in zip(values, estimates[self.current]))
            ]
        best = min(eligible, key=lambda name: costs[name] + transition_us) if eligible else self.current
        if best == self.current:
            self.pending, self.count = None, 0
        else:
            self.count = self.count + 1 if self.pending == best else 1
            self.pending = best
            if self.count >= self.confirmations:
                self.current, self.pending, self.count = best, None, 0
        applied_transition = transition_us if self.current != original else 0.0
        return dict(path=self.current, reason=reason,
                    predicted_us=costs[self.current] + applied_transition,
                    predicted_path_us=costs[self.current], transition_us=applied_transition,
                    predicted_selected_reads=demand, demand_interval=(lower, upper), estimates=costs)
