"""Half-hourly electricity demand forecasting for the Victorian region of the NEM."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nem-forecast")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
