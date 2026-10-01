"""Named historical and hypothetical stress scenarios.

Stress is deliberately separate from probabilistic VaR: a scenario carries no
probability. A scenario applies explicit ``instrument_id`` simple-return shocks
to the current as-of currency exposures:

``scenario_pnl[i] = x[i] * q[i]`` and ``scenario_loss[i] = -scenario_pnl[i]``.

Historical scenarios derive their shocks from observed simple returns over an
explicit session range using the compounding helper
``cumulative_shock[i] = product(1 + simple_return[i, d]) - 1``. Published
``(scenario_id, scenario_version)`` pairs are immutable and content-hashed;
loading a catalog re-verifies every hash. Factor shocks and silent proxies are
out of scope.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from historical_asset_risk.contracts import (
    STRESS_CATALOG_SCHEMA_ID,
    STRESS_SCENARIO_SCHEMA_VERSION,
)
from historical_asset_risk.portfolio_risk import compound_simple_returns

_SCENARIO_KINDS = frozenset({"historical", "hypothetical"})


class StressScenarioError(ValueError):
    """Raised when a stress scenario or catalog violates its contract."""


@dataclass(frozen=True)
class InstrumentShock:
    """One instrument's simple-return shock in a scenario."""

    instrument_id: str
    simple_return_shock: float


@dataclass(frozen=True)
class StressScenario:
    """An immutable, content-hashed named shock to a book."""

    scenario_id: str
    scenario_version: str
    name: str
    kind: str
    description: str
    shocks: tuple[InstrumentShock, ...]
    content_hash: str
    session_start: str | None = None
    session_end: str | None = None


def canonical_scenario_payload(
    *,
    scenario_id: str,
    scenario_version: str,
    name: str,
    kind: str,
    description: str,
    shocks: Sequence[InstrumentShock],
    session_start: str | None = None,
    session_end: str | None = None,
) -> dict[str, Any]:
    """Return the canonical dict a scenario's ``content_hash`` is taken over."""
    return {
        "scenario_id": scenario_id,
        "scenario_version": scenario_version,
        "name": name,
        "kind": kind,
        "description": description,
        "session_start": session_start,
        "session_end": session_end,
        "shocks": [
            {
                "instrument_id": shock.instrument_id,
                "simple_return_shock": shock.simple_return_shock,
            }
            for shock in sorted(shocks, key=lambda item: item.instrument_id)
        ],
    }


def scenario_content_hash(payload: Mapping[str, Any]) -> str:
    """Return the deterministic ``sha256:`` identity of a scenario payload."""
    canonical = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _parse_shocks(raw: Any, scenario_id: str) -> tuple[InstrumentShock, ...]:
    if not isinstance(raw, list) or not raw:
        raise StressScenarioError(
            f"Scenario {scenario_id!r} must list at least one shock."
        )
    shocks: list[InstrumentShock] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise StressScenarioError(
                f"Scenario {scenario_id!r} has a malformed shock entry."
            )
        instrument_id = str(entry.get("instrument_id", "")).strip()
        if not instrument_id:
            raise StressScenarioError(
                f"Scenario {scenario_id!r} has a shock with no instrument_id."
            )
        if instrument_id in seen:
            raise StressScenarioError(
                f"Scenario {scenario_id!r} shocks {instrument_id!r} more than once."
            )
        seen.add(instrument_id)
        try:
            shock_value = float(entry["simple_return_shock"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StressScenarioError(
                f"Scenario {scenario_id!r} shock for {instrument_id!r} is not "
                "a finite number."
            ) from exc
        if not math.isfinite(shock_value):
            raise StressScenarioError(
                f"Scenario {scenario_id!r} shock for {instrument_id!r} is not finite."
            )
        if shock_value < -1.0:
            raise StressScenarioError(
                f"Scenario {scenario_id!r} shock for {instrument_id!r} is "
                f"{shock_value!r}; a simple return cannot fall below -1."
            )
        shocks.append(InstrumentShock(instrument_id, shock_value))
    return tuple(shocks)


def load_stress_catalog(path: Path) -> tuple[StressScenario, ...]:
    """Load and hash-verify a stress-scenario catalog JSON file."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StressScenarioError(
            f"Could not read stress catalog {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise StressScenarioError("The stress catalog must be a JSON object.")
    if document.get("schema_id") != STRESS_CATALOG_SCHEMA_ID:
        raise StressScenarioError(
            f"Stress catalog schema_id must be {STRESS_CATALOG_SCHEMA_ID!r}."
        )
    if document.get("schema_version") != STRESS_SCENARIO_SCHEMA_VERSION:
        raise StressScenarioError(
            f"Stress catalog schema_version must be {STRESS_SCENARIO_SCHEMA_VERSION!r}."
        )
    raw_scenarios = document.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise StressScenarioError("The stress catalog lists no scenarios.")

    scenarios: list[StressScenario] = []
    identities: set[tuple[str, str]] = set()
    for raw in raw_scenarios:
        if not isinstance(raw, dict):
            raise StressScenarioError("Each scenario must be a JSON object.")
        scenario_id = str(raw.get("scenario_id", "")).strip()
        scenario_version = str(raw.get("scenario_version", "")).strip()
        if not scenario_id or not scenario_version:
            raise StressScenarioError(
                "Every scenario needs a scenario_id and scenario_version."
            )
        identity = (scenario_id, scenario_version)
        if identity in identities:
            raise StressScenarioError(
                f"Duplicate scenario identity {scenario_id!r} "
                f"version {scenario_version!r}."
            )
        identities.add(identity)
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in _SCENARIO_KINDS:
            raise StressScenarioError(
                f"Scenario {scenario_id!r} kind must be 'historical' or 'hypothetical'."
            )
        name = str(raw.get("name", "")).strip() or scenario_id
        description = str(raw.get("description", "")).strip()
        session_start = raw.get("session_start")
        session_end = raw.get("session_end")
        session_start = str(session_start) if session_start is not None else None
        session_end = str(session_end) if session_end is not None else None
        shocks = _parse_shocks(raw.get("shocks"), scenario_id)

        payload = canonical_scenario_payload(
            scenario_id=scenario_id,
            scenario_version=scenario_version,
            name=name,
            kind=kind,
            description=description,
            shocks=shocks,
            session_start=session_start,
            session_end=session_end,
        )
        expected_hash = scenario_content_hash(payload)
        declared_hash = str(raw.get("content_hash", "")).strip()
        if not declared_hash:
            raise StressScenarioError(
                f"Scenario {scenario_id!r} is missing content_hash."
            )
        if declared_hash != expected_hash:
            raise StressScenarioError(
                f"Scenario {scenario_id!r} version {scenario_version!r} content "
                f"hash does not match its declared identity ({declared_hash} != "
                f"{expected_hash})."
            )
        scenarios.append(
            StressScenario(
                scenario_id=scenario_id,
                scenario_version=scenario_version,
                name=name,
                kind=kind,
                description=description,
                shocks=shocks,
                content_hash=expected_hash,
                session_start=session_start,
                session_end=session_end,
            )
        )
    return tuple(scenarios)


def historical_shock_vector(
    simple_returns: pd.DataFrame,
    *,
    session_start: str | pd.Timestamp,
    session_end: str | pd.Timestamp,
) -> dict[str, float]:
    """Compounded ``product(1 + r) - 1`` shock per instrument over a session range."""
    compounded = compound_simple_returns(
        simple_returns, start=session_start, end=session_end
    )
    return {str(key): float(value) for key, value in compounded.items()}


@dataclass(frozen=True)
class StressContribution:
    """One instrument's linear contribution to a stress result."""

    instrument_id: str
    currency_exposure: float
    simple_return_shock: float
    scenario_pnl: float
    scenario_loss: float


@dataclass(frozen=True)
class StressResult:
    """Deterministic P&L of a book under one named scenario."""

    scenario_id: str
    scenario_version: str
    name: str
    kind: str
    content_hash: str
    scenario_pnl: float
    scenario_loss: float
    contributions: tuple[StressContribution, ...]


def apply_stress(
    scenario: StressScenario,
    currency_exposures: Mapping[str, float],
) -> StressResult:
    """Apply a scenario's shocks to signed base-currency exposures.

    Every held instrument (an entry in ``currency_exposures``) must be shocked
    by the scenario; a held instrument with no shock fails. Scenario shocks for
    instruments the book does not hold contribute zero and are retained for
    audit.
    """
    shock_by_id = {
        shock.instrument_id: shock.simple_return_shock for shock in scenario.shocks
    }
    exposures = {str(key): float(value) for key, value in currency_exposures.items()}
    missing = sorted(
        instrument_id for instrument_id in exposures if instrument_id not in shock_by_id
    )
    if missing:
        raise StressScenarioError(
            f"Scenario {scenario.scenario_id!r} does not shock held instrument(s): "
            + ", ".join(missing)
            + "."
        )
    for instrument_id, exposure in exposures.items():
        if not math.isfinite(exposure):
            raise StressScenarioError(
                f"Currency exposure for {instrument_id!r} is not finite."
            )

    contributions: list[StressContribution] = []
    for instrument_id in sorted(set(exposures) | set(shock_by_id)):
        exposure = exposures.get(instrument_id, 0.0)
        shock = shock_by_id.get(instrument_id, 0.0)
        pnl = exposure * shock
        contributions.append(
            StressContribution(
                instrument_id=instrument_id,
                currency_exposure=exposure,
                simple_return_shock=shock,
                scenario_pnl=pnl,
                scenario_loss=-pnl,
            )
        )
    scenario_pnl = math.fsum(item.scenario_pnl for item in contributions)
    return StressResult(
        scenario_id=scenario.scenario_id,
        scenario_version=scenario.scenario_version,
        name=scenario.name,
        kind=scenario.kind,
        content_hash=scenario.content_hash,
        scenario_pnl=scenario_pnl,
        scenario_loss=-scenario_pnl,
        contributions=tuple(contributions),
    )


__all__ = [
    "STRESS_CATALOG_SCHEMA_ID",
    "STRESS_SCENARIO_SCHEMA_VERSION",
    "InstrumentShock",
    "StressContribution",
    "StressResult",
    "StressScenario",
    "StressScenarioError",
    "apply_stress",
    "canonical_scenario_payload",
    "historical_shock_vector",
    "load_stress_catalog",
    "scenario_content_hash",
]
