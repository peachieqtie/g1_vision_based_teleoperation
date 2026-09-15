"""One-Euro filter for smoothing noisy keypoints."""

from __future__ import annotations
import numpy as np


class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1 - alpha) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._x_prev = None

    def reset(self):
        """Forget all history. Episode reset must call this, or the first frame
        of a new episode is blended with the last frame of the previous one."""
        self._x = _LowPass()
        self._dx = _LowPass()
        self._x_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self._x_prev is None:
            self._x_prev = x
            self._x.y = x
            return x
        dx = (x - self._x_prev) * self.freq
        edx = self._dx(dx, self._alpha(self.d_cutoff, self.freq))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        y = self._x(x, self._alpha(cutoff, self.freq))
        self._x_prev = x
        return y


class KeypointFilter:
    def __init__(self, keypoint_indices, freq=30.0, min_cutoff=1.0, beta=0.007):
        self._filters = {
            idx: [OneEuroFilter(freq, min_cutoff, beta) for _ in range(3)]
            for idx in keypoint_indices
        }

    def __call__(self, keypoints):
        out = list(keypoints)
        for idx, filts in self._filters.items():
            p = np.asarray(keypoints[idx], float)
            if np.any(np.isnan(p)):
                continue
            out[idx] = np.array([filts[i](p[i]) for i in range(3)])
        return out

    def reset(self):
        """Reset every per-keypoint filter (see OneEuroFilter.reset).

        `_filters` maps keypoint index -> a list of THREE filters, one per axis.
        """
        for filts in self._filters.values():
            for f in filts:
                f.reset()
