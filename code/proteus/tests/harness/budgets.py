"""Load performance budgets from budgets.toml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


_BUDGETS_PATH = Path(__file__).parent / "budgets.toml"


def load_budgets(size_class: str = "small") -> dict[str, Any]:
    """Load budget limits for the given data-size class."""
    with open(_BUDGETS_PATH, "rb") as f:
        data = tomllib.load(f)
    if size_class not in data:
        raise KeyError(
            f"Unknown size class {size_class!r}; "
            f"available: {list(data.keys())}"
        )
    return dict(data[size_class])
