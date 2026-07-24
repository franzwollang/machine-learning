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
    components: Sequence[FadedComponent]
    support: Support

    @property
    def weights(self) -> np.ndarray:
        raw = np.array([max(float(comp.weight), _EPS) for comp in self.components], dtype=float)
        return raw / raw.sum()

    def density(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        in_support = self.support.contains(arr).astype(float)
        u = self.support.uniform_density
        out = np.zeros(arr.shape[0], dtype=float)
        for weight, comp in zip(self.weights, self.components, strict=True):
            lam = comp.fade_weight(arr)
            out += weight * (lam * comp.density(arr) + (1.0 - lam) * u * in_support)
        return out

    def proposal_density(self, x: np.ndarray, proposal_signal_fraction: float) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        in_support = self.support.contains(arr).astype(float)
        gauss_mix = np.zeros(arr.shape[0], dtype=float)
        for weight, comp in zip(self.weights, self.components, strict=True):
            gauss_mix += weight * comp.density(arr)
        return proposal_signal_fraction * gauss_mix + (1.0 - proposal_signal_fraction) * self.support.uniform_density * in_support

    def draw_proposals(
        self,
        n: int,
        rng: np.random.Generator,
        proposal_signal_fraction: float,
    ) -> np.ndarray:
        proposals = np.empty((n, self.support.dim), dtype=float)
        use_signal = rng.random(n) < proposal_signal_fraction
        signal_count = int(use_signal.sum())
        if signal_count > 0:
            indices = rng.choice(len(self.components), size=signal_count, p=self.weights)
            positions = np.where(use_signal)[0]
            for comp_idx, comp in enumerate(self.components):
                mask = indices == comp_idx
                count = int(mask.sum())
                if count > 0:
                    proposals[positions[mask]] = comp.sample(count, rng)
        if signal_count < n:
            positions = np.where(~use_signal)[0]
            proposals[positions] = self.support.sample_uniform(n - signal_count, rng)
        return proposals


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
    """Rejection-sample exactly from the faded density with fixed-envelope restarts."""
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
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
