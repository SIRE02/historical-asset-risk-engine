"""Validated configuration loading for analysis runs."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from historical_asset_risk.estimation import (
    DEFAULT_DOWNSIDE_TARGET,
    DEFAULT_OBSERVATIONS_PER_YEAR,
    DEFAULT_QUANTILE_METHOD,
    DEFAULT_QUANTILES,
    DEFAULT_ROLLING_WINDOW,
    resolve_rolling_min_observations,
    validate_finite_number,
    validate_positive_integer,
    validate_quantile_method,
    validate_quantiles,
)

DEFAULT_CONFIGURATION: dict[str, Any] = {
    "provider": "yahoo",
    "tickers": ["SPY", "QQQ", "TLT", "GLD"],
    "start_date": "2020-01-01",
    "end_date": "2025-01-01",
    "rolling_window": DEFAULT_ROLLING_WINDOW,
    "rolling_min_observations": None,
    "observations_per_year": DEFAULT_OBSERVATIONS_PER_YEAR,
    "quantiles": list(DEFAULT_QUANTILES),
    "quantile_method": DEFAULT_QUANTILE_METHOD,
    "downside_target": DEFAULT_DOWNSIDE_TARGET,
    "output_dir": "outputs",
    "csv_path": None,
    "instrument_registry_path": None,
    "positions_path": None,
    "cash_path": None,
    "positions_history_path": None,
    "cash_history_path": None,
}


@dataclass(frozen=True)
class AnalysisConfig:
    """Complete validated configuration for one analysis run."""

    provider: str = "yahoo"
    tickers: tuple[str, ...] = ("SPY", "QQQ", "TLT", "GLD")
    start_date: str = "2020-01-01"
    end_date: str = "2025-01-01"
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    rolling_min_observations: int | None = None
    observations_per_year: int = DEFAULT_OBSERVATIONS_PER_YEAR
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    quantile_method: str = DEFAULT_QUANTILE_METHOD
    downside_target: float = DEFAULT_DOWNSIDE_TARGET
    output_dir: Path = Path("outputs")
    csv_path: Path | None = None
    instrument_registry_path: Path | None = None
    positions_path: Path | None = None
    cash_path: Path | None = None
    positions_history_path: Path | None = None
    cash_history_path: Path | None = None

    def __post_init__(self) -> None:
        provider = str(self.provider).strip().lower()
        if provider not in {"yahoo", "csv"}:
            raise ValueError("PROVIDER must be either 'yahoo' or 'csv'.")

        tickers_value = self.tickers
        if isinstance(tickers_value, (str, bytes)):
            tickers_value = tuple(str(tickers_value).split(","))
        symbols = tuple(
            dict.fromkeys(str(ticker).strip().upper() for ticker in tickers_value)
        )
        symbols = tuple(symbol for symbol in symbols if symbol)
        if len(symbols) < 2:
            raise ValueError("At least two unique ticker symbols are required.")

        try:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "START_DATE and END_DATE must use YYYY-MM-DD format."
            ) from exc
        if start >= end:
            raise ValueError("START_DATE must be earlier than END_DATE.")

        rolling_min_observations = resolve_rolling_min_observations(
            self.rolling_window, self.rolling_min_observations
        )
        validate_positive_integer(self.observations_per_year, "OBSERVATIONS_PER_YEAR")
        quantiles = validate_quantiles(self.quantiles)
        quantile_method = validate_quantile_method(self.quantile_method)
        downside_target = validate_finite_number(
            self.downside_target, "DOWNSIDE_TARGET"
        )

        output_dir = Path(self.output_dir).expanduser()
        if not str(output_dir).strip():
            raise ValueError("OUTPUT_DIR must not be empty.")
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"OUTPUT_DIR is not a directory: {output_dir}")
        csv_path = (
            Path(self.csv_path).expanduser() if self.csv_path is not None else None
        )
        if provider == "csv":
            if csv_path is None:
                raise ValueError("CSV_PATH is required when PROVIDER is 'csv'.")
            if not csv_path.is_file():
                raise ValueError(
                    f"CSV_PATH does not identify a readable file: {csv_path}"
                )
        portfolio_paths = {
            "INSTRUMENT_REGISTRY_PATH": self.instrument_registry_path,
            "POSITIONS_PATH": self.positions_path,
            "CASH_PATH": self.cash_path,
        }
        supplied_portfolio_paths = {
            name: value for name, value in portfolio_paths.items() if value is not None
        }
        if supplied_portfolio_paths and len(supplied_portfolio_paths) != len(
            portfolio_paths
        ):
            missing = sorted(set(portfolio_paths) - set(supplied_portfolio_paths))
            raise ValueError(
                "Portfolio analysis requires all three paths; missing "
                + ", ".join(missing)
                + "."
            )
        normalized_portfolio_paths: dict[str, Path | None] = {}
        for name, value in portfolio_paths.items():
            path = Path(value).expanduser() if value is not None else None
            if path is not None and not path.is_file():
                raise ValueError(f"{name} does not identify a readable file: {path}")
            normalized_portfolio_paths[name] = path

        history_paths = {
            "POSITIONS_HISTORY_PATH": self.positions_history_path,
            "CASH_HISTORY_PATH": self.cash_history_path,
        }
        supplied_history = {
            name: value for name, value in history_paths.items() if value is not None
        }
        if supplied_history:
            if len(supplied_history) != len(history_paths):
                missing = sorted(set(history_paths) - set(supplied_history))
                raise ValueError(
                    "Proxy realized P&L requires both history paths; missing "
                    + ", ".join(missing)
                    + "."
                )
            if not supplied_portfolio_paths:
                raise ValueError(
                    "POSITIONS_HISTORY_PATH and CASH_HISTORY_PATH require the "
                    "primary portfolio paths as well."
                )
        normalized_history_paths: dict[str, Path | None] = {}
        for name, value in history_paths.items():
            path = Path(value).expanduser() if value is not None else None
            if path is not None and not path.is_file():
                raise ValueError(f"{name} does not identify a readable file: {path}")
            normalized_history_paths[name] = path

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "tickers", symbols)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "quantile_method", quantile_method)
        object.__setattr__(self, "downside_target", downside_target)
        object.__setattr__(self, "rolling_min_observations", rolling_min_observations)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "csv_path", csv_path)
        object.__setattr__(
            self,
            "instrument_registry_path",
            normalized_portfolio_paths["INSTRUMENT_REGISTRY_PATH"],
        )
        object.__setattr__(
            self, "positions_path", normalized_portfolio_paths["POSITIONS_PATH"]
        )
        object.__setattr__(self, "cash_path", normalized_portfolio_paths["CASH_PATH"])
        object.__setattr__(
            self,
            "positions_history_path",
            normalized_history_paths["POSITIONS_HISTORY_PATH"],
        )
        object.__setattr__(
            self, "cash_history_path", normalized_history_paths["CASH_HISTORY_PATH"]
        )

    @property
    def portfolio_enabled(self) -> bool:
        """Whether the complete Phase 3 input set is configured."""
        return self.instrument_registry_path is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        values = asdict(self)
        values["tickers"] = list(self.tickers)
        values["quantiles"] = list(self.quantiles)
        values["output_dir"] = str(self.output_dir)
        values["csv_path"] = str(self.csv_path) if self.csv_path is not None else None
        values["instrument_registry_path"] = (
            str(self.instrument_registry_path)
            if self.instrument_registry_path is not None
            else None
        )
        values["positions_path"] = (
            str(self.positions_path) if self.positions_path is not None else None
        )
        values["cash_path"] = (
            str(self.cash_path) if self.cash_path is not None else None
        )
        values["positions_history_path"] = (
            str(self.positions_history_path)
            if self.positions_history_path is not None
            else None
        )
        values["cash_history_path"] = (
            str(self.cash_history_path)
            if self.cash_history_path is not None
            else None
        )
        return values


def _read_config_file(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".toml":
            with path.open("rb") as stream:
                loaded = tomllib.load(stream)
        else:
            raise ValueError("Configuration files must use .toml or .json.")
    except OSError as exc:
        raise ValueError(f"Could not read configuration file {path}: {exc}") from exc
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid configuration file {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError("The configuration file must contain an object/table.")
    if "analysis" in loaded:
        loaded = loaded["analysis"]
        if not isinstance(loaded, dict):
            raise ValueError("The 'analysis' configuration must be an object/table.")
    return loaded


def load_configuration(
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AnalysisConfig:
    """Merge defaults, an optional TOML/JSON file, and command-line overrides."""
    values = dict(DEFAULT_CONFIGURATION)
    if config_path is not None:
        path = Path(config_path)
        file_values = _read_config_file(path)
        unknown = sorted(set(file_values) - set(DEFAULT_CONFIGURATION))
        if unknown:
            raise ValueError(f"Unknown configuration option(s): {', '.join(unknown)}")
        values.update(file_values)

    if overrides:
        unknown = sorted(set(overrides) - set(DEFAULT_CONFIGURATION))
        if unknown:
            raise ValueError(f"Unknown configuration override(s): {', '.join(unknown)}")
        values.update(
            {key: value for key, value in overrides.items() if value is not None}
        )

    tickers = values["tickers"]
    if isinstance(tickers, str):
        values["tickers"] = tuple(part for part in tickers.split(","))
    elif isinstance(tickers, Sequence):
        values["tickers"] = tuple(tickers)
    else:
        raise ValueError("TICKERS must be a list or a comma-separated string.")
    quantiles = values["quantiles"]
    if isinstance(quantiles, str):
        values["quantiles"] = tuple(part for part in quantiles.split(","))
    elif isinstance(quantiles, Sequence):
        values["quantiles"] = tuple(quantiles)
    else:
        raise ValueError("QUANTILES must be a list or a comma-separated string.")
    values["output_dir"] = Path(values["output_dir"])
    if values.get("csv_path") is not None:
        values["csv_path"] = Path(values["csv_path"])
    for path_name in (
        "instrument_registry_path",
        "positions_path",
        "cash_path",
        "positions_history_path",
        "cash_history_path",
    ):
        if values.get(path_name) is not None:
            values[path_name] = Path(values[path_name])
    return AnalysisConfig(**values)
