"""Shared exact faded-density primitives for synthetic datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

_EPS = 1e-12


def _ball_volume(dim: int, radius: float) -> float:
    from math import gamma as _gamma, pi as _pi

    return (_pi ** (dim / 2.0) / _gamma(dim / 2.0 + 1.0)) * (radius ** dim)


def _sphere_surface_area(dim: int, radius: np.ndarray) -> np.ndarray:
    from math import gamma as _gamma, pi as _pi

    coeff = 2.0 * (_pi ** (dim / 2.0)) / _gamma(dim / 2.0)
    return coeff * np.power(np.maximum(radius, _EPS), dim - 1)


def arc_primitive(value: float | np.ndarray) -> np.ndarray:
    """Antiderivative of ``sqrt(1 + t^2)`` (Swiss-roll arc-length element)."""
    arr = np.asarray(value, dtype=float)
    return 0.5 * (arr * np.sqrt(1.0 + arr * arr) + np.arcsinh(arr))


def gaussian_pdf(x: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic Gaussian density at each row of ``x``."""
    dim = mean.shape[0]
    diff = np.asarray(x, dtype=float) - mean[None, :]
    maha_sq = (diff * diff).sum(axis=1) / max(sigma * sigma, _EPS)
    log_norm = -0.5 * dim * np.log(2.0 * np.pi) - dim * np.log(max(sigma, _EPS))
    return np.exp(log_norm - 0.5 * maha_sq)


def lambda_from_distance(
    distance: np.ndarray,
    sigma: float,
    transition_radius: float,
) -> np.ndarray:
    """Fade weight lambda(d) = exp(-d^2 / (2 (r sigma)^2))."""
    scale = max(float(transition_radius) * float(sigma), _EPS)
    return np.exp(-0.5 * np.square(np.asarray(distance, dtype=float) / scale))


class Support(Protocol):
    @property
    def dim(self) -> int:
        ...

    @property
    def uniform_density(self) -> float:
        ...

    def contains(self, x: np.ndarray) -> np.ndarray:
        ...

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        ...

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class SupportBall:
    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")

    @property
    def dim(self) -> int:
        return int(self.center.shape[0])

    @property
    def uniform_density(self) -> float:
        return 1.0 / max(_ball_volume(self.dim, self.radius), _EPS)

    def contains(self, x: np.ndarray) -> np.ndarray:
        diff = np.asarray(x, dtype=float) - self.center[None, :]
        return np.linalg.norm(diff, axis=1) <= self.radius

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        vec = rng.normal(size=(n, self.dim))
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / np.maximum(norms, _EPS)
        radii = self.radius * rng.random(n) ** (1.0 / self.dim)
        return self.center[None, :] + vec * radii[:, None]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.center - self.radius, self.center + self.radius


@dataclass(frozen=True)
class SupportBox:
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        lo = np.asarray(self.lo, dtype=float)
        hi = np.asarray(self.hi, dtype=float)
        if lo.shape != hi.shape:
            raise ValueError("lo and hi must share a shape")
        if np.any(hi <= lo):
            raise ValueError("all hi coordinates must exceed lo coordinates")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def dim(self) -> int:
        return int(self.lo.shape[0])

    @property
    def uniform_density(self) -> float:
        return 1.0 / max(float(np.prod(self.hi - self.lo)), _EPS)

    def contains(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return np.all((arr >= self.lo[None, :]) & (arr <= self.hi[None, :]), axis=1)

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lo, self.hi, size=(n, self.dim))

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lo, self.hi

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        *,
        padding_fraction: float = 0.05,
        min_padding: float = 0.05,
        extra_padding: float = 0.0,
    ) -> "SupportBox":
        arr = np.asarray(points, dtype=float)
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        span = hi - lo
        padding = np.maximum(padding_fraction * span, min_padding)
        padding = np.maximum(padding, float(extra_padding))
        return cls(lo=lo - padding, hi=hi + padding)


class FadedComponent(Protocol):
    @property
    def weight(self) -> float:
        ...

    @property
    def sigma(self) -> float:
        ...

    @property
    def transition_radius(self) -> float:
        ...

    def density(self, x: np.ndarray) -> np.ndarray:
        ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        ...

    def distance(self, x: np.ndarray) -> np.ndarray:
        ...

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class GaussianFadedComponent:
    center: np.ndarray
    sigma: float
    transition_radius: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))

    def density(self, x: np.ndarray) -> np.ndarray:
        return gaussian_pdf(np.asarray(x, dtype=float), self.center, self.sigma)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(n, self.center.shape[0])) * self.sigma + self.center[None, :]

    def distance(self, x: np.ndarray) -> np.ndarray:
        diff = np.asarray(x, dtype=float) - self.center[None, :]
        return np.linalg.norm(diff, axis=1)

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(self.distance(x), self.sigma, self.transition_radius)


@dataclass(frozen=True)
class KernelMixtureFadedComponent:
    anchors: np.ndarray
    sigma: float
    transition_radius: float
    weight: float = 1.0
    anchor_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchors, dtype=float)
        if anchors.ndim != 2 or anchors.shape[0] == 0:
            raise ValueError("anchors must have shape (m, d) with m > 0")
        object.__setattr__(self, "anchors", anchors)
        if self.anchor_weights is not None:
            weights = np.asarray(self.anchor_weights, dtype=float)
            if weights.shape != (anchors.shape[0],):
                raise ValueError("anchor_weights must match number of anchors")
            weights = weights / max(weights.sum(), _EPS)
            object.__setattr__(self, "anchor_weights", weights)

    @property
    def dim(self) -> int:
        return int(self.anchors.shape[1])

    def _weights(self) -> np.ndarray:
        if self.anchor_weights is None:
            return np.full(self.anchors.shape[0], 1.0 / self.anchors.shape[0], dtype=float)
        return self.anchor_weights

    def density(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        weights = self._weights()
        out = np.zeros(arr.shape[0], dtype=float)
        chunk = max(32, min(256, self.anchors.shape[0]))
        for start in range(0, self.anchors.shape[0], chunk):
            stop = min(start + chunk, self.anchors.shape[0])
            anchor_chunk = self.anchors[start:stop]
            diff = arr[:, None, :] - anchor_chunk[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            log_norm = -0.5 * self.dim * np.log(2.0 * np.pi) - self.dim * np.log(max(self.sigma, _EPS))
            out += np.exp(log_norm - 0.5 * dist_sq / max(self.sigma * self.sigma, _EPS)) @ weights[start:stop]
        return out

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        weights = self._weights()
        indices = rng.choice(self.anchors.shape[0], size=n, p=weights)
        return self.anchors[indices] + rng.normal(scale=self.sigma, size=(n, self.dim))

    def distance(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        best = np.full(arr.shape[0], np.inf, dtype=float)
        chunk = max(32, min(256, self.anchors.shape[0]))
        for start in range(0, self.anchors.shape[0], chunk):
            stop = min(start + chunk, self.anchors.shape[0])
            diff = arr[:, None, :] - self.anchors[None, start:stop, :]
            dist_sq = np.sum(diff * diff, axis=2)
            best = np.minimum(best, dist_sq.min(axis=1))
        return np.sqrt(best)

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(self.distance(x), self.sigma, self.transition_radius)


@dataclass(frozen=True)
class CircleFadedComponent:
    radius: float
    sigma: float
    transition_radius: float
    center: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.ndim != 1 or center.shape[0] < 2:
            raise ValueError("center must be a 1D vector with ambient dim >= 2")
        object.__setattr__(self, "center", center)

    @property
    def dim(self) -> int:
        return int(self.center.shape[0])

    def density(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        rel = arr - self.center[None, :]
        rho = np.linalg.norm(rel[:, :2], axis=1)
        radial_residual = rho - self.radius
        radial_pdf = np.exp(-0.5 * np.square(radial_residual / max(self.sigma, _EPS)))
        radial_pdf /= np.sqrt(2.0 * np.pi) * max(self.sigma, _EPS)
        planar_density = radial_pdf / np.maximum(2.0 * np.pi * rho, _EPS)
        if self.dim <= 2:
            return planar_density
        extras = rel[:, 2:]
        extra_norm_sq = np.sum(extras * extras, axis=1)
        log_norm = -0.5 * extras.shape[1] * np.log(2.0 * np.pi) - extras.shape[1] * np.log(max(self.sigma, _EPS))
        extra_pdf = np.exp(log_norm - 0.5 * extra_norm_sq / max(self.sigma * self.sigma, _EPS))
        return planar_density * extra_pdf

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        radial = self.radius + rng.normal(scale=self.sigma, size=n)
        out = np.zeros((n, self.dim), dtype=float)
        out[:, 0] = radial * np.cos(theta)
        out[:, 1] = radial * np.sin(theta)
        if self.dim > 2:
            out[:, 2:] = rng.normal(scale=self.sigma, size=(n, self.dim - 2))
        return out + self.center[None, :]

    def distance(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        rel = arr - self.center[None, :]
        rho = np.linalg.norm(rel[:, :2], axis=1)
        radial_residual = rho - self.radius
        if self.dim <= 2:
            return np.abs(radial_residual)
        return np.sqrt(np.square(radial_residual) + np.sum(rel[:, 2:] * rel[:, 2:], axis=1))

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(self.distance(x), self.sigma, self.transition_radius)


@dataclass(frozen=True)
class BimodalCircleFadedComponent:
    """Circle with a von Mises angular density (two modes, connected support).

    The support is a single circle.  Angular density is an equal mixture of
    von Mises concentrations at ``mode_angles``.  This is the Hartigan-valley
    scene the uniform circle is not: connected, but ``{p >= lambda}``
    disconnects into two arcs.
    """

    radius: float
    sigma: float
    transition_radius: float
    center: np.ndarray
    kappa: float = 3.0
    mode_angles: tuple[float, float] = (0.0, float(np.pi))
    weight: float = 1.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.ndim != 1 or center.shape[0] < 2:
            raise ValueError("center must be a 1D vector with ambient dim >= 2")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        object.__setattr__(self, "center", center)

    @property
    def dim(self) -> int:
        return int(self.center.shape[0])

    def _theta(self, x: np.ndarray) -> np.ndarray:
        rel = np.asarray(x, dtype=float) - self.center[None, :]
        return np.arctan2(rel[:, 1], rel[:, 0])

    def _angular_pdf(self, theta: np.ndarray) -> np.ndarray:
        from scipy.special import i0

        z = 2.0 * np.pi * float(i0(self.kappa))
        pdf = np.zeros(theta.shape[0], dtype=float)
        for mu in self.mode_angles:
            pdf += np.exp(self.kappa * np.cos(theta - mu)) / z
        return pdf / float(len(self.mode_angles))

    def density(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        rel = arr - self.center[None, :]
        rho = np.linalg.norm(rel[:, :2], axis=1)
        radial_residual = rho - self.radius
        radial_pdf = np.exp(-0.5 * np.square(radial_residual / max(self.sigma, _EPS)))
        radial_pdf /= np.sqrt(2.0 * np.pi) * max(self.sigma, _EPS)
        planar = radial_pdf / np.maximum(rho, _EPS) * self._angular_pdf(self._theta(arr))
        if self.dim <= 2:
            return planar
        extras = rel[:, 2:]
        extra_norm_sq = np.sum(extras * extras, axis=1)
        log_norm = (
            -0.5 * extras.shape[1] * np.log(2.0 * np.pi)
            - extras.shape[1] * np.log(max(self.sigma, _EPS))
        )
        extra_pdf = np.exp(
            log_norm - 0.5 * extra_norm_sq / max(self.sigma * self.sigma, _EPS)
        )
        return planar * extra_pdf

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        which = rng.integers(0, len(self.mode_angles), size=n)
        theta = np.empty(n, dtype=float)
        for i, mu in enumerate(self.mode_angles):
            mask = which == i
            if mask.any():
                theta[mask] = rng.vonmises(mu, self.kappa, size=int(mask.sum()))
        radial = self.radius + rng.normal(scale=self.sigma, size=n)
        out = np.zeros((n, self.dim), dtype=float)
        out[:, 0] = radial * np.cos(theta)
        out[:, 1] = radial * np.sin(theta)
        if self.dim > 2:
            out[:, 2:] = rng.normal(scale=self.sigma, size=(n, self.dim - 2))
        return out + self.center[None, :]

    def distance(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        rel = arr - self.center[None, :]
        rho = np.linalg.norm(rel[:, :2], axis=1)
        radial_residual = rho - self.radius
        if self.dim <= 2:
            return np.abs(radial_residual)
        return np.sqrt(
            np.square(radial_residual) + np.sum(rel[:, 2:] * rel[:, 2:], axis=1)
        )

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(self.distance(x), self.sigma, self.transition_radius)

    def mode_labels(self, x: np.ndarray) -> np.ndarray:
        """Nearest von Mises mode in angle; does not apply the fade threshold."""

        theta = self._theta(x)
        deltas = [
            np.abs(np.arctan2(np.sin(theta - mu), np.cos(theta - mu)))
            for mu in self.mode_angles
        ]
        return np.argmin(np.stack(deltas, axis=0), axis=0).astype(int)


@dataclass(frozen=True)
class TorusSurfaceFadedComponent:
    """Area-uniform torus surface with Gaussian fade in normal directions.

    ``rotation`` maps local torus coordinates into the first three world
    coordinates.  Remaining ambient coordinates are independent normal
    directions.  Unlike a kernel-anchor approximation, this component is
    continuous around both torus angles and therefore has no artificial
    lattice of density modes.
    """

    major_radius: float
    minor_radius: float
    sigma: float
    transition_radius: float
    center: np.ndarray
    rotation: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        rotation = np.asarray(self.rotation, dtype=float)
        if center.ndim != 1 or center.shape[0] < 3:
            raise ValueError("center must be 1D with ambient dim >= 3")
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10):
            raise ValueError("rotation must be orthogonal")
        if self.major_radius <= 0.0 or self.minor_radius <= 0.0:
            raise ValueError("torus radii must be positive")
        if self.minor_radius >= self.major_radius:
            raise ValueError("minor_radius must be smaller than major_radius")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "rotation", rotation)

    @property
    def dim(self) -> int:
        return int(self.center.shape[0])

    def _local(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rel = np.asarray(x, dtype=float) - self.center[None, :]
        local3 = rel[:, :3] @ self.rotation
        return local3, rel[:, 3:]

    def density(self, x: np.ndarray) -> np.ndarray:
        local, extras = self._local(x)
        planar_radius = np.linalg.norm(local[:, :2], axis=1)
        tube_radius = np.sqrt(
            np.square(planar_radius - self.major_radius)
            + np.square(local[:, 2])
        )
        residual = tube_radius - self.minor_radius
        radial_pdf = np.exp(
            -0.5 * np.square(residual / max(self.sigma, _EPS))
        )
        radial_pdf /= np.sqrt(2.0 * np.pi) * max(self.sigma, _EPS)
        level_area = (
            4.0 * np.pi**2 * self.major_radius
            * np.maximum(tube_radius, _EPS)
        )
        density = radial_pdf / level_area
        if extras.shape[1] == 0:
            return density
        norm_sq = np.sum(extras * extras, axis=1)
        log_norm = (
            -0.5 * extras.shape[1] * np.log(2.0 * np.pi)
            - extras.shape[1] * np.log(max(self.sigma, _EPS))
        )
        return density * np.exp(
            log_norm - 0.5 * norm_sq / max(self.sigma**2, _EPS)
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        radial = self.minor_radius + rng.normal(scale=self.sigma, size=n)
        radial = np.maximum(radial, _EPS)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        phi = np.empty(n, dtype=float)
        pending = np.arange(n)
        while pending.size:
            proposal = rng.uniform(0.0, 2.0 * np.pi, size=pending.size)
            numer = self.major_radius + radial[pending] * np.cos(proposal)
            denom = self.major_radius + radial[pending]
            accept = rng.random(pending.size) < (numer / denom)
            phi[pending[accept]] = proposal[accept]
            pending = pending[~accept]

        local = np.zeros((n, 3), dtype=float)
        ring_radius = self.major_radius + radial * np.cos(phi)
        local[:, 0] = ring_radius * np.cos(theta)
        local[:, 1] = ring_radius * np.sin(theta)
        local[:, 2] = radial * np.sin(phi)
        out = np.zeros((n, self.dim), dtype=float)
        out[:, :3] = local @ self.rotation.T
        if self.dim > 3:
            out[:, 3:] = rng.normal(
                scale=self.sigma, size=(n, self.dim - 3),
            )
        return out + self.center[None, :]

    def distance(self, x: np.ndarray) -> np.ndarray:
        local, extras = self._local(x)
        planar_radius = np.linalg.norm(local[:, :2], axis=1)
        tube_radius = np.sqrt(
            np.square(planar_radius - self.major_radius)
            + np.square(local[:, 2])
        )
        residual_sq = np.square(tube_radius - self.minor_radius)
        if extras.shape[1]:
            residual_sq += np.sum(extras * extras, axis=1)
        return np.sqrt(residual_sq)

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(
            self.distance(x), self.sigma, self.transition_radius,
        )


_SWISS_ROLL_GRID = 4096
_SWISS_ROLL_NEWTON_STEPS = 2
_SWISS_ROLL_NEWTON_STEP_CAP = 0.05


@dataclass(frozen=True)
class SwissRollSurfaceFadedComponent:
    """Area-uniform Swiss-roll sheet with Gaussian fade in normal directions.

    The spiral sheet is
    ``(t cos t, h, t sin t) / scale`` for ``t in [t_min, t_max]`` and
    ``h in [0, height]``, with ``scale = t_max`` by default.  Unlike a
    kernel-anchor lattice, this component is continuous along both
    parameters and therefore has no artificial ridge-line modes.
    Remaining ambient coordinates (``ambient_dim > 3``) are independent
    normal directions, matching ``TorusSurfaceFadedComponent``.
    """

    t_min: float
    t_max: float
    height: float
    sigma: float
    transition_radius: float
    ambient_dim: int = 3
    weight: float = 1.0
    scale: float | None = None

    def __post_init__(self) -> None:
        if self.t_max <= self.t_min:
            raise ValueError("t_max must exceed t_min")
        if self.height <= 0.0:
            raise ValueError("height must be positive")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.ambient_dim < 3:
            raise ValueError("ambient_dim must be at least 3")
        scale = float(self.t_max if self.scale is None else self.scale)
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        object.__setattr__(self, "scale", scale)

        t_grid = np.linspace(self.t_min, self.t_max, num=_SWISS_ROLL_GRID)
        object.__setattr__(self, "_t_grid", t_grid)
        object.__setattr__(self, "_spiral_grid", self._spiral_xz(t_grid))
        prim = arc_primitive(t_grid)
        cdf = (prim - prim[0]) / max(float(prim[-1] - prim[0]), _EPS)
        object.__setattr__(self, "_cdf", cdf)
        area = (
            float(self.height)
            / (scale * scale)
            * float(arc_primitive(self.t_max) - arc_primitive(self.t_min))
        )
        object.__setattr__(self, "_surface_area", max(area, _EPS))

    @property
    def dim(self) -> int:
        return int(self.ambient_dim)

    @property
    def surface_area(self) -> float:
        return float(self._surface_area)

    def _spiral_xz(self, t: np.ndarray) -> np.ndarray:
        scale = float(self.scale)
        return np.stack([t * np.cos(t), t * np.sin(t)], axis=-1) / scale

    def _unit_normal_xz(self, t: np.ndarray) -> np.ndarray:
        length = np.sqrt(1.0 + t * t)
        return np.stack(
            [-(np.sin(t) + t * np.cos(t)), np.cos(t) - t * np.sin(t)],
            axis=-1,
        ) / np.maximum(length, _EPS)[:, None]

    def _project_t(self, q: np.ndarray) -> np.ndarray:
        """Nearest parameter ``t`` of the xz spiral for each query ``q``."""
        n = q.shape[0]
        grid = self._spiral_grid
        t_grid = self._t_grid
        best_d2 = np.full(n, np.inf, dtype=float)
        best_t = np.zeros(n, dtype=float)
        chunk = 256
        for start in range(0, t_grid.shape[0], chunk):
            stop = min(start + chunk, t_grid.shape[0])
            diff = q[:, None, :] - grid[None, start:stop, :]
            dist_sq = np.sum(diff * diff, axis=2)
            local_idx = dist_sq.argmin(axis=1)
            local_d2 = dist_sq[np.arange(n), local_idx]
            better = local_d2 < best_d2
            best_d2[better] = local_d2[better]
            best_t[better] = t_grid[start:stop][local_idx[better]]

        t = best_t
        scale = float(self.scale)
        for _ in range(_SWISS_ROLL_NEWTON_STEPS):
            ct = np.cos(t)
            st = np.sin(t)
            gamma = np.stack([t * ct, t * st], axis=1) / scale
            g1 = np.stack([ct - t * st, st + t * ct], axis=1) / scale
            g2 = np.stack([-2.0 * st - t * ct, 2.0 * ct - t * st], axis=1) / scale
            resid = q - gamma
            fp = -np.sum(resid * g1, axis=1)
            fpp = np.sum(g1 * g1, axis=1) - np.sum(resid * g2, axis=1)
            step = np.zeros_like(t)
            good = fpp > _EPS
            step[good] = fp[good] / fpp[good]
            step = np.clip(step, -_SWISS_ROLL_NEWTON_STEP_CAP, _SWISS_ROLL_NEWTON_STEP_CAP)
            t = np.clip(t - step, self.t_min, self.t_max)
        return t

    def density(self, x: np.ndarray) -> np.ndarray:
        dist = self.distance(x)
        sigma = max(self.sigma, _EPS)
        codim = self.ambient_dim - 2
        normalizer = self._surface_area * (2.0 * np.pi * sigma * sigma) ** (0.5 * codim)
        return np.exp(-0.5 * np.square(dist / sigma)) / max(normalizer, _EPS)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(n)
        t = np.interp(u, self._cdf, self._t_grid)
        h = rng.uniform(0.0, self.height, size=n)
        scale = float(self.scale)
        normal_xz = self._unit_normal_xz(t)
        offset = rng.normal(scale=self.sigma, size=n)
        out = np.zeros((n, self.ambient_dim), dtype=float)
        out[:, 0] = t * np.cos(t) / scale + offset * normal_xz[:, 0]
        out[:, 1] = h / scale
        out[:, 2] = t * np.sin(t) / scale + offset * normal_xz[:, 1]
        if self.ambient_dim > 3:
            out[:, 3:] = rng.normal(
                scale=self.sigma, size=(n, self.ambient_dim - 3),
            )
        return out

    def distance(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        q = arr[:, [0, 2]]
        t = self._project_t(q)
        gamma = self._spiral_xz(t)
        xz_res_sq = np.sum((q - gamma) * (q - gamma), axis=1)
        y_hi = self.height / float(self.scale)
        y_clamped = np.clip(arr[:, 1], 0.0, y_hi)
        d_sq = xz_res_sq + np.square(arr[:, 1] - y_clamped)
        if self.ambient_dim > 3:
            extras = arr[:, 3:]
            d_sq = d_sq + np.sum(extras * extras, axis=1)
        return np.sqrt(d_sq)

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(
            self.distance(x), self.sigma, self.transition_radius,
        )


@dataclass(frozen=True)
class AxisAlignedSheetFadedComponent:
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    ambient_dim: int
    sigma: float
    transition_radius: float
    offset: np.ndarray | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.ambient_dim < 2:
            raise ValueError("ambient_dim must be at least 2")
        u0, u1 = self.u_range
        v0, v1 = self.v_range
        if u1 <= u0 or v1 <= v0:
            raise ValueError("sheet ranges must be increasing")
        if self.offset is None:
            object.__setattr__(self, "offset", np.zeros(self.ambient_dim, dtype=float))
        else:
            offset = np.asarray(self.offset, dtype=float)
            if offset.shape != (self.ambient_dim,):
                raise ValueError("offset must match ambient_dim")
            object.__setattr__(self, "offset", offset)

    @property
    def resolved_offset(self) -> np.ndarray:
        offset = self.offset
        assert offset is not None
        return offset

    def density(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float) - self.resolved_offset[None, :]
        inside = (
            (arr[:, 0] >= self.u_range[0]) & (arr[:, 0] <= self.u_range[1]) &
            (arr[:, 1] >= self.v_range[0]) & (arr[:, 1] <= self.v_range[1])
        ).astype(float)
        area = max((self.u_range[1] - self.u_range[0]) * (self.v_range[1] - self.v_range[0]), _EPS)
        if self.ambient_dim <= 2:
            return inside / area
        normals = arr[:, 2:]
        norm_sq = np.sum(normals * normals, axis=1)
        log_norm = -0.5 * normals.shape[1] * np.log(2.0 * np.pi) - normals.shape[1] * np.log(max(self.sigma, _EPS))
        return inside * np.exp(log_norm - 0.5 * norm_sq / max(self.sigma * self.sigma, _EPS)) / area

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros((n, self.ambient_dim), dtype=float)
        out[:, 0] = rng.uniform(self.u_range[0], self.u_range[1], size=n)
        out[:, 1] = rng.uniform(self.v_range[0], self.v_range[1], size=n)
        if self.ambient_dim > 2:
            out[:, 2:] = rng.normal(scale=self.sigma, size=(n, self.ambient_dim - 2))
        return out + self.resolved_offset[None, :]

    def distance(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float) - self.resolved_offset[None, :]
        du = np.where(arr[:, 0] < self.u_range[0], self.u_range[0] - arr[:, 0], np.where(arr[:, 0] > self.u_range[1], arr[:, 0] - self.u_range[1], 0.0))
        dv = np.where(arr[:, 1] < self.v_range[0], self.v_range[0] - arr[:, 1], np.where(arr[:, 1] > self.v_range[1], arr[:, 1] - self.v_range[1], 0.0))
        if self.ambient_dim <= 2:
            return np.sqrt(du * du + dv * dv)
        return np.sqrt(du * du + dv * dv + np.sum(arr[:, 2:] * arr[:, 2:], axis=1))

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(self.distance(x), self.sigma, self.transition_radius)


@dataclass(frozen=True)
class AxisAlignedBoxFadedComponent:
    """Solid axis-aligned box signal spanning the first ``len(lo)`` axes.

    Uniform density inside ``[lo_j, hi_j]`` along each spanned axis, with an
    isotropic Gaussian fade of width ``sigma`` in the remaining
    ``ambient_dim - len(lo)`` normal directions.  With ``len(lo) == ambient_dim``
    the component is a uniform solid box with no normal directions --- an
    intrinsic ``ambient_dim``-manifold (the 3D "box" of the classic GNG
    manifold-zoo benchmark, OPEN_ISSUES #26).
    """

    lo: tuple[float, ...]
    hi: tuple[float, ...]
    ambient_dim: int
    sigma: float
    transition_radius: float
    offset: np.ndarray | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        lo = np.asarray(self.lo, dtype=float)
        hi = np.asarray(self.hi, dtype=float)
        if lo.shape != hi.shape or lo.ndim != 1:
            raise ValueError("lo and hi must be 1D and share a shape")
        if np.any(hi <= lo):
            raise ValueError("all hi coordinates must exceed lo coordinates")
        if lo.shape[0] > self.ambient_dim:
            raise ValueError("box spans more axes than ambient_dim")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        if self.offset is None:
            object.__setattr__(self, "offset", np.zeros(self.ambient_dim, dtype=float))
        else:
            offset = np.asarray(self.offset, dtype=float)
            if offset.shape != (self.ambient_dim,):
                raise ValueError("offset must match ambient_dim")
            object.__setattr__(self, "offset", offset)

    @property
    def box_dim(self) -> int:
        return int(np.asarray(self.lo).shape[0])

    @property
    def resolved_offset(self) -> np.ndarray:
        offset = self.offset
        assert offset is not None
        return offset

    def density(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float) - self.resolved_offset[None, :]
        k = self.box_dim
        spanned = arr[:, :k]
        inside = np.all(
            (spanned >= np.asarray(self.lo)[None, :])
            & (spanned <= np.asarray(self.hi)[None, :]),
            axis=1,
        ).astype(float)
        volume = max(float(np.prod(np.asarray(self.hi) - np.asarray(self.lo))), _EPS)
        if self.ambient_dim <= k:
            return inside / volume
        normals = arr[:, k:]
        norm_sq = np.sum(normals * normals, axis=1)
        log_norm = -0.5 * normals.shape[1] * np.log(2.0 * np.pi) - normals.shape[1] * np.log(max(self.sigma, _EPS))
        return inside * np.exp(log_norm - 0.5 * norm_sq / max(self.sigma * self.sigma, _EPS)) / volume

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = self.box_dim
        out = np.zeros((n, self.ambient_dim), dtype=float)
        out[:, :k] = rng.uniform(np.asarray(self.lo), np.asarray(self.hi), size=(n, k))
        if self.ambient_dim > k:
            out[:, k:] = rng.normal(scale=self.sigma, size=(n, self.ambient_dim - k))
        return out + self.resolved_offset[None, :]

    def distance(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float) - self.resolved_offset[None, :]
        k = self.box_dim
        spanned = arr[:, :k]
        lo = np.asarray(self.lo)[None, :]
        hi = np.asarray(self.hi)[None, :]
        below = np.where(spanned < lo, lo - spanned, 0.0)
        above = np.where(spanned > hi, spanned - hi, 0.0)
        d_sq = np.sum(below * below + above * above, axis=1)
        if self.ambient_dim > k:
            normals = arr[:, k:]
            d_sq = d_sq + np.sum(normals * normals, axis=1)
        return np.sqrt(d_sq)

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(self.distance(x), self.sigma, self.transition_radius)


@dataclass(frozen=True)
class AxisAlignedSegmentFadedComponent:
    t_range: tuple[float, float]
    ambient_dim: int
    sigma: float
    transition_radius: float
    offset: np.ndarray | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.ambient_dim < 1:
            raise ValueError("ambient_dim must be positive")
        t0, t1 = self.t_range
        if t1 <= t0:
            raise ValueError("segment range must be increasing")
        if self.offset is None:
            object.__setattr__(self, "offset", np.zeros(self.ambient_dim, dtype=float))
        else:
            offset = np.asarray(self.offset, dtype=float)
            if offset.shape != (self.ambient_dim,):
                raise ValueError("offset must match ambient_dim")
            object.__setattr__(self, "offset", offset)

    @property
    def resolved_offset(self) -> np.ndarray:
        offset = self.offset
        assert offset is not None
        return offset

    def density(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float) - self.resolved_offset[None, :]
        inside = ((arr[:, 0] >= self.t_range[0]) & (arr[:, 0] <= self.t_range[1])).astype(float)
        length = max(self.t_range[1] - self.t_range[0], _EPS)
        if self.ambient_dim <= 1:
            return inside / length
        normals = arr[:, 1:]
        norm_sq = np.sum(normals * normals, axis=1)
        log_norm = -0.5 * normals.shape[1] * np.log(2.0 * np.pi) - normals.shape[1] * np.log(max(self.sigma, _EPS))
        return inside * np.exp(log_norm - 0.5 * norm_sq / max(self.sigma * self.sigma, _EPS)) / length

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros((n, self.ambient_dim), dtype=float)
        out[:, 0] = rng.uniform(self.t_range[0], self.t_range[1], size=n)
        if self.ambient_dim > 1:
            out[:, 1:] = rng.normal(scale=self.sigma, size=(n, self.ambient_dim - 1))
        return out + self.resolved_offset[None, :]

    def distance(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float) - self.resolved_offset[None, :]
        dt = np.where(arr[:, 0] < self.t_range[0], self.t_range[0] - arr[:, 0], np.where(arr[:, 0] > self.t_range[1], arr[:, 0] - self.t_range[1], 0.0))
        if self.ambient_dim <= 1:
            return np.abs(dt)
        return np.sqrt(dt * dt + np.sum(arr[:, 1:] * arr[:, 1:], axis=1))

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(self.distance(x), self.sigma, self.transition_radius)


@dataclass(frozen=True)
class SphereShellFadedComponent:
    radius: float
    base_dim: int
    sigma: float
    transition_radius: float
    center: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.ndim != 1 or center.shape[0] < self.base_dim:
            raise ValueError("center must be 1D and at least base_dim long")
        object.__setattr__(self, "center", center)
        if self.base_dim < 2:
            raise ValueError("base_dim must be at least 2")

    @property
    def dim(self) -> int:
        return int(self.center.shape[0])

    def density(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float) - self.center[None, :]
        sphere_coords = arr[:, :self.base_dim]
        rho = np.linalg.norm(sphere_coords, axis=1)
        radial = rho - self.radius
        radial_pdf = np.exp(-0.5 * np.square(radial / max(self.sigma, _EPS)))
        radial_pdf /= np.sqrt(2.0 * np.pi) * max(self.sigma, _EPS)
        base_density = radial_pdf / np.maximum(_sphere_surface_area(self.base_dim, rho), _EPS)
        if self.dim <= self.base_dim:
            return base_density
        extras = arr[:, self.base_dim:]
        norm_sq = np.sum(extras * extras, axis=1)
        log_norm = -0.5 * extras.shape[1] * np.log(2.0 * np.pi) - extras.shape[1] * np.log(max(self.sigma, _EPS))
        return base_density * np.exp(log_norm - 0.5 * norm_sq / max(self.sigma * self.sigma, _EPS))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        directions = rng.normal(size=(n, self.base_dim))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norms, _EPS)
        radial = self.radius + rng.normal(scale=self.sigma, size=n)
        out = np.zeros((n, self.dim), dtype=float)
        out[:, :self.base_dim] = directions * radial[:, None]
        if self.dim > self.base_dim:
            out[:, self.base_dim:] = rng.normal(scale=self.sigma, size=(n, self.dim - self.base_dim))
        return out + self.center[None, :]

    def distance(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float) - self.center[None, :]
        rho = np.linalg.norm(arr[:, :self.base_dim], axis=1)
        radial = rho - self.radius
        if self.dim <= self.base_dim:
            return np.abs(radial)
        return np.sqrt(np.square(radial) + np.sum(arr[:, self.base_dim:] * arr[:, self.base_dim:], axis=1))

    def fade_weight(self, x: np.ndarray) -> np.ndarray:
        return lambda_from_distance(self.distance(x), self.sigma, self.transition_radius)


@dataclass(frozen=True)
class FadedMixture:
    """Faded signal-plus-tissue density on a compact support.

    Default (``tissue_mass is None``) uses the legacy fade-balanced floor
    ``Σ_c w_c [λ_c f_c + (1-λ_c) u]``.  That form puts roughly equal mass on
    the signal and the uniform box floor, so λ<0.5 labels land near 46–49%
    regardless of the generators' historical ``tissue_fraction`` padding knob.

    When ``tissue_mass`` is set in ``[0, 1)``, the density is the honest
    two-region mixture that places mass ``tissue_mass`` on the λ<0.5
    (background) region and ``1 - tissue_mass`` on the λ≥0.5 region, so the
    expected label tissue fraction matches the request.
    """

    components: Sequence[FadedComponent]
    support: Support
    tissue_mass: float | None = None

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("components must be non-empty")
        if self.tissue_mass is not None:
            mass = float(self.tissue_mass)
            if not 0.0 <= mass < 1.0:
                raise ValueError("tissue_mass must lie in [0, 1)")
            object.__setattr__(self, "tissue_mass", mass)
        object.__setattr__(self, "_region_norm_cache", {})

    @property
    def weights(self) -> np.ndarray:
        raw = np.array([max(float(comp.weight), _EPS) for comp in self.components], dtype=float)
        return raw / raw.sum()

    def lambda_max(self, x: np.ndarray) -> np.ndarray:
        """Pointwise max fade weight across components (label tissue iff < 0.5)."""
        arr = np.asarray(x, dtype=float)
        best = np.zeros(arr.shape[0], dtype=float)
        for comp in self.components:
            best = np.maximum(best, comp.fade_weight(arr))
        return best

    def signal_density(self, x: np.ndarray) -> np.ndarray:
        """Mixture of component densities without the uniform tissue floor."""
        arr = np.asarray(x, dtype=float)
        out = np.zeros(arr.shape[0], dtype=float)
        for weight, comp in zip(self.weights, self.components, strict=True):
            out += weight * comp.density(arr)
        return out

    def _region_normalizers(self, *, n_mc: int = 20000) -> tuple[float, float]:
        """MC estimates of ∫_{λ≥0.5} signal_density and ∫_{λ<0.5} u."""
        cache = self._region_norm_cache
        if cache:
            return float(cache["zs"]), float(cache["zt"])
        rng = np.random.default_rng(0)
        # Importance: draw from the component mixture (~∫ signal_density ≈ 1).
        sig = self.draw_signal_proposals(n_mc, rng)
        in_support = self.support.contains(sig)
        zs = float(np.mean((self.lambda_max(sig) >= 0.5) & in_support))
        uni = self.support.sample_uniform(n_mc, rng)
        zt = float(np.mean(self.lambda_max(uni) < 0.5))
        zs = max(zs, _EPS)
        zt = max(zt, _EPS)
        cache["zs"] = zs
        cache["zt"] = zt
        return zs, zt

    def density(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        in_support = self.support.contains(arr)
        u = self.support.uniform_density
        if self.tissue_mass is None:
            # Legacy fade-balanced floor (≈ half the cloud labelled tissue).
            mask = in_support.astype(float)
            out = np.zeros(arr.shape[0], dtype=float)
            for weight, comp in zip(self.weights, self.components, strict=True):
                lam = comp.fade_weight(arr)
                out += weight * (lam * comp.density(arr) + (1.0 - lam) * u * mask)
            return out

        mass = float(self.tissue_mass)
        gauss = self.signal_density(arr)
        lam = self.lambda_max(arr)
        is_tissue = in_support & (lam < 0.5)
        is_signal = in_support & (lam >= 0.5)
        zs, zt = self._region_normalizers()
        out = np.zeros(arr.shape[0], dtype=float)
        out[is_signal] = (1.0 - mass) * gauss[is_signal] / zs
        out[is_tissue] = mass * u / zt
        return out

    def proposal_density(self, x: np.ndarray, proposal_signal_fraction: float) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        in_support = self.support.contains(arr).astype(float)
        gauss_mix = self.signal_density(arr)
        if self.tissue_mass is None:
            frac = float(proposal_signal_fraction)
        else:
            # Match the honest mass mixture: tissue_mass weight on uniform.
            frac = 1.0 - float(self.tissue_mass)
        return frac * gauss_mix + (1.0 - frac) * self.support.uniform_density * in_support

    def draw_signal_proposals(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` proposals from the component mixture alone."""
        proposals = np.empty((n, self.support.dim), dtype=float)
        if n <= 0:
            return proposals
        indices = rng.choice(len(self.components), size=n, p=self.weights)
        for comp_idx, comp in enumerate(self.components):
            mask = indices == comp_idx
            count = int(mask.sum())
            if count > 0:
                proposals[mask] = comp.sample(count, rng)
        return proposals

    def draw_proposals(
        self,
        n: int,
        rng: np.random.Generator,
        proposal_signal_fraction: float,
    ) -> np.ndarray:
        proposals = np.empty((n, self.support.dim), dtype=float)
        if self.tissue_mass is None:
            use_signal = rng.random(n) < proposal_signal_fraction
        else:
            use_signal = rng.random(n) >= float(self.tissue_mass)
        signal_count = int(use_signal.sum())
        if signal_count > 0:
            proposals[use_signal] = self.draw_signal_proposals(signal_count, rng)
        if signal_count < n:
            proposals[~use_signal] = self.support.sample_uniform(n - signal_count, rng)
        return proposals

    def sample_region(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        tissue: bool,
        max_rounds: int = 200,
    ) -> np.ndarray:
        """Sample ``n`` points from the λ-threshold tissue or signal region."""
        n = int(n)
        if n <= 0:
            return np.empty((0, self.support.dim), dtype=float)
        accepted: list[np.ndarray] = []
        remaining = n
        for _ in range(max_rounds):
            if remaining <= 0:
                break
            batch_n = max(128, 8 * remaining)
            if tissue:
                props = self.support.sample_uniform(batch_n, rng)
            else:
                props = self.draw_signal_proposals(batch_n, rng)
            in_support = self.support.contains(props)
            lam = self.lambda_max(props)
            keep = in_support & ((lam < 0.5) if tissue else (lam >= 0.5))
            if keep.any():
                kept = props[keep][:remaining]
                accepted.append(kept)
                remaining -= int(kept.shape[0])
        if remaining > 0:
            kind = "tissue" if tissue else "signal"
            raise RuntimeError(
                f"failed to sample {n} {kind}-region points "
                f"(short by {remaining}); check support / fade geometry"
            )
        return np.vstack(accepted)[:n]


def tissue_mass_metadata(
    *,
    tissue_fraction: float,
    tissue_mass: float | None,
    labels: np.ndarray,
) -> dict[str, float | int | str | None]:
    """Standard requested-vs-actual tissue/signal metadata for faded generators.

    ``tissue_fraction`` is the historical support-box *padding* knob — it does
    not set background mass.  ``tissue_mass`` is the honest mass fraction
    (``None`` = legacy fade-balanced floor).  Counts and actual mass come from
    λ-threshold labels (``< 0`` = tissue).
    """
    lab = np.asarray(labels)
    n = int(lab.shape[0])
    tissue_count = int(np.sum(lab < 0))
    signal_count = int(n - tissue_count)
    tissue_mass_actual = float(tissue_count / max(n, 1))
    if tissue_mass is None:
        tissue_mass_requested: float | None = None
        tissue_count_requested: int | None = None
        signal_count_requested: int | None = None
        mode = "legacy_fade_balanced"
    else:
        mass = float(tissue_mass)
        tissue_mass_requested = mass
        tissue_count_requested = min(max(int(np.round(mass * n)), 0), n)
        signal_count_requested = int(n - tissue_count_requested)
        mode = "requested_mass"
    return {
        "tissue_fraction_requested": float(tissue_fraction),
        "tissue_fraction_role": "support_box_padding",
        "tissue_fraction_actual": tissue_mass_actual,
        "tissue_mass_requested": tissue_mass_requested,
        "tissue_mass_actual": tissue_mass_actual,
        "tissue_mass_mode": mode,
        "signal_count_requested": signal_count_requested,
        "signal_count_actual": signal_count,
        "tissue_count_requested": tissue_count_requested,
        "tissue_count_actual": tissue_count,
        # Short aliases match sample_faded_mixture region-conditional keys.
        "signal_count": signal_count,
        "tissue_count": tissue_count,
    }


def assign_labels_by_lambda(
    points: np.ndarray,
    components: Sequence[FadedComponent],
    *,
    label_offsets: list[int] | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Assign each point to its dominant component or ``-1`` for tissue."""
    arr = np.asarray(points, dtype=float)
    labels = np.full(arr.shape[0], -1, dtype=int)
    best_lambda = np.zeros(arr.shape[0], dtype=float)
    if label_offsets is None:
        label_offsets = list(range(len(components)))
    for comp, label in zip(components, label_offsets, strict=True):
        lam = comp.fade_weight(arr)
        better = lam > best_lambda
        labels[better] = int(label)
        best_lambda[better] = lam[better]
    labels[best_lambda < threshold] = -1
    return labels


def sample_faded_mixture(
    mixture: FadedMixture,
    n_samples: int,
    rng: np.random.Generator,
    *,
    proposal_signal_fraction: float = 0.7,
    max_rounds: int = 500,
    max_restarts: int = 6,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Rejection-sample exactly from the faded density with fixed-envelope restarts.

    When ``mixture.tissue_mass`` is set, samples are drawn region-conditionally
    so the λ<0.5 label fraction matches the requested mass (up to rounding).
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if mixture.tissue_mass is not None:
        mass = float(mixture.tissue_mass)
        n_tissue = int(np.round(mass * n_samples))
        n_tissue = min(max(n_tissue, 0), n_samples)
        n_signal = n_samples - n_tissue
        tissue_pts = mixture.sample_region(n_tissue, rng, tissue=True)
        signal_pts = mixture.sample_region(n_signal, rng, tissue=False)
        if n_tissue and n_signal:
            points = np.vstack([tissue_pts, signal_pts])
        elif n_tissue:
            points = tissue_pts
        else:
            points = signal_pts
        perm = rng.permutation(n_samples)
        return points[perm], {
            "acceptance_rate": 1.0,
            "total_proposal_draws": int(n_samples),
            "proposal_m_bound": 1.0,
            "sampler_restarts": 0,
            "tissue_mass_requested": mass,
            "tissue_count": int(n_tissue),
            "signal_count": int(n_signal),
        }

    bound = 0.0
    total_drawn = 0

    for restart in range(max_restarts):
        accepted: list[np.ndarray] = []
        remaining = n_samples
        violated = False
        drawn_this_attempt = 0
        current_bound = bound if bound > 0.0 else None

        for _ in range(max_rounds):
            if remaining <= 0:
                break
            batch_n = max(512, 8 * remaining)
            proposals = mixture.draw_proposals(batch_n, rng, proposal_signal_fraction)
            drawn_this_attempt += batch_n

            target = mixture.density(proposals)
            proposal = mixture.proposal_density(proposals, proposal_signal_fraction)
            ratio = np.where(proposal > _EPS, target / proposal, 0.0)
            batch_max = float(np.max(ratio)) if ratio.size else 1.0
            if current_bound is None:
                current_bound = 1.1 * batch_max
            elif batch_max > current_bound:
                bound = 1.1 * batch_max
                violated = True
                break

            accept_prob = np.minimum(ratio / max(current_bound, _EPS), 1.0)
            keep = rng.random(batch_n) < accept_prob
            if keep.any():
                kept = proposals[keep][:remaining]
                accepted.append(kept)
                remaining -= kept.shape[0]

        total_drawn += drawn_this_attempt
        if violated:
            continue
        if remaining <= 0:
            bound = float(current_bound) if current_bound is not None else 0.0
            points = np.vstack(accepted)[:n_samples]
            return points, {
                "acceptance_rate": float(n_samples / max(total_drawn, 1)),
                "total_proposal_draws": int(total_drawn),
                "proposal_m_bound": float(bound),
                "sampler_restarts": int(restart),
            }

    raise RuntimeError(
        f"failed to sample exact faded mixture after {max_restarts} restarts; "
        f"final envelope bound={bound:.4f}"
    )
