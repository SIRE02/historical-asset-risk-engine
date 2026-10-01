"""Deterministic, versioned market-session validation for portfolio books."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache

from historical_asset_risk.contracts import PortfolioCalendarError

CALENDAR_SOURCE = "historical-asset-risk built-in XNYS session calendar"
CALENDAR_VERSION = "xnys-1990-2035.2"
SUPPORTED_CALENDAR_ID = "XNYS"
SUPPORTED_MARKET_TIMEZONE = "America/New_York"
_FIRST_YEAR = 1990
_LAST_YEAR = 2035

_SPECIAL_CLOSURES = {
    date(1994, 4, 27),  # President Nixon funeral
    date(2001, 9, 11),
    date(2001, 9, 12),
    date(2001, 9, 13),
    date(2001, 9, 14),
    date(2004, 6, 11),  # President Reagan funeral
    date(2007, 1, 2),  # President Ford funeral
    date(2012, 10, 29),  # Hurricane Sandy
    date(2012, 10, 30),
    date(2018, 12, 5),  # President George H. W. Bush funeral
    date(2025, 1, 9),  # President Carter funeral
}


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    candidate = date(year, month, 1)
    candidate += timedelta(days=(weekday - candidate.weekday()) % 7)
    return candidate + timedelta(weeks=occurrence - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        candidate = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        candidate = date(year, month + 1, 1) - timedelta(days=1)
    return candidate - timedelta(days=(candidate.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Return Gregorian Easter using the anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


# Cached because a portfolio run checks every aligned price date.
@cache
def _holidays(year: int) -> frozenset[date]:
    holidays = {
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    new_years_day = date(year, 1, 1)
    # Unlike the other fixed-date holidays, the NYSE does not observe New
    # Year's Day on the preceding Friday when January 1 falls on a Saturday.
    if new_years_day.weekday() != 5:
        holidays.add(_observed(new_years_day))
    if year >= 1998:
        holidays.add(_nth_weekday(year, 1, 0, 3))  # MLK Day
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))  # Juneteenth
    return frozenset(holidays)


@dataclass(frozen=True)
class MarketCalendar:
    """Resolved offline market calendar contract."""

    calendar_id: str
    source: str = CALENDAR_SOURCE
    version: str = CALENDAR_VERSION
    market_timezone: str = SUPPORTED_MARKET_TIMEZONE

    def is_session(self, session_date: date) -> bool:
        if not _FIRST_YEAR <= session_date.year <= _LAST_YEAR:
            raise PortfolioCalendarError(
                f"{self.calendar_id} calendar {self.version} supports years "
                f"{_FIRST_YEAR} through {_LAST_YEAR}; received {session_date}."
            )
        return (
            session_date.weekday() < 5
            and session_date
            not in (_holidays(session_date.year) | _holidays(session_date.year + 1))
            and session_date not in _SPECIAL_CLOSURES
        )

    def require_consecutive_sessions(self, observed: Iterable[date]) -> None:
        """Fail unless ``observed`` dates are consecutive sessions on this calendar.

        A daily return between two observed dates is a one-session return only
        when no session lies between them. Complete-case alignment already
        fails on a hole in one instrument; a session missing from every
        instrument leaves no hole to see, so only the calendar can catch it.
        """
        ordered = sorted(set(observed))
        non_sessions = [day for day in ordered if not self.is_session(day)]
        if non_sessions:
            examples = ", ".join(day.isoformat() for day in non_sessions[:3])
            raise PortfolioCalendarError(
                f"Adjusted prices are dated on non-{self.calendar_id} sessions: "
                f"{examples}."
            )
        gaps: list[str] = []
        for previous, current in zip(ordered, ordered[1:], strict=False):
            missing: list[date] = []
            candidate = self.next_session(previous)
            while candidate < current:
                missing.append(candidate)
                candidate = self.next_session(candidate)
            if missing:
                skipped = ", ".join(day.isoformat() for day in missing[:3])
                gaps.append(
                    f"{previous.isoformat()} to {current.isoformat()} "
                    f"(missing {skipped})"
                )
        if gaps:
            raise PortfolioCalendarError(
                f"Adjusted prices skip {self.calendar_id} session(s): "
                f"{'; '.join(gaps[:3])}. A return across them would span more "
                "than one session but be treated as one day."
            )

    def next_session(self, session_date: date) -> date:
        """Return the first valid session strictly after ``session_date``.

        Used for one-day proxy realization: a realization after the close on
        date ``t`` is aligned to the next valid session on this calendar, not
        necessarily the next civil day.
        """
        candidate = session_date + timedelta(days=1)
        # A single week never contains more than the weekend plus a short run of
        # holidays; the supported-year guard in ``is_session`` bounds the search.
        for _ in range(10):
            if self.is_session(candidate):
                return candidate
            candidate += timedelta(days=1)
        raise PortfolioCalendarError(
            f"No {self.calendar_id} session found within 10 days after "
            f"{session_date.isoformat()}."
        )


def resolve_market_calendar(
    calendar_id: str, required_version: str | None = None
) -> MarketCalendar:
    """Resolve the exact supported calendar identity and optional version."""
    normalized = str(calendar_id).strip().upper()
    if normalized != SUPPORTED_CALENDAR_ID:
        raise PortfolioCalendarError(
            f"Unsupported market_calendar_id {calendar_id!r}; "
            f"supported value is {SUPPORTED_CALENDAR_ID!r}."
        )
    if required_version is not None and required_version != CALENDAR_VERSION:
        raise PortfolioCalendarError(
            f"Unsupported calendar version {required_version!r}; "
            f"{SUPPORTED_CALENDAR_ID} requires {CALENDAR_VERSION!r}."
        )
    return MarketCalendar(normalized)


__all__ = [
    "CALENDAR_SOURCE",
    "CALENDAR_VERSION",
    "SUPPORTED_MARKET_TIMEZONE",
    "MarketCalendar",
    "resolve_market_calendar",
]
