"""
Pure helpers for MBTA data: prediction timing/sorting and alert classification.

Kept free of Flask and network calls so they can be unit-tested directly.
"""

from datetime import timedelta
from dateutil import parser

# Effects that describe station accessibility rather than train/bus service.
ACCESSIBILITY_EFFECTS = {
    "ELEVATOR_CLOSURE", "ESCALATOR_CLOSURE", "ACCESS_ISSUE", "FACILITY_ISSUE",
    "PARKING_ISSUE", "PARKING_CLOSURE", "BIKE_ISSUE", "STATION_ISSUE",
}

# GTFS route_type -> mode key used by the frontend.
ROUTE_TYPE_MODES = {0: "subway", 1: "subway", 2: "commuter_rail", 3: "bus", 4: "ferry"}

# Predictions that are this far in the past are dropped from arrival boards.
DEPARTED_GRACE = timedelta(seconds=60)


def effective_time(attrs):
    """The time a prediction is displayed and sorted by: arrival, else departure."""
    return attrs.get("arrival_time") or attrs.get("departure_time")


def event_kind(attrs):
    """'arrival' when the train arrives, 'departure' when only a departure is known
    (e.g. trains leaving a terminal like Oak Grove)."""
    return "arrival" if attrs.get("arrival_time") else "departure"


def parse_time(value):
    try:
        return parser.isoparse(value) if value else None
    except (ValueError, TypeError):
        return None


def sort_by_time(items, key="time"):
    """Stable sort by an ISO timestamp field; items without a time go last."""
    def sort_key(item):
        dt = parse_time(item.get(key))
        return (0, dt.timestamp()) if dt else (1, 0)

    return sorted(items, key=sort_key)


def is_departed(dt, now):
    return dt is not None and dt < now - DEPARTED_GRACE


def is_cancelled(attrs):
    return (attrs.get("schedule_relationship") or "").upper() in {"CANCELLED", "SKIPPED", "NO_DATA"}


def alert_route_types(attrs):
    return sorted({e["route_type"] for e in attrs.get("informed_entity", []) if "route_type" in e})


def alert_modes(attrs):
    return sorted({ROUTE_TYPE_MODES[t] for t in alert_route_types(attrs) if t in ROUTE_TYPE_MODES})


def alert_is_active(attrs, now):
    """True when `now` falls inside one of the alert's active periods."""
    periods = attrs.get("active_period") or []
    if not periods:
        return True
    for period in periods:
        start = parse_time(period.get("start"))
        end = parse_time(period.get("end"))
        if (start is None or start <= now) and (end is None or now <= end):
            return True
    return False


def alert_payload(alert, now):
    """Normalize a raw MBTA alert into the shape the frontend uses."""
    attrs = alert.get("attributes", {})
    entities = attrs.get("informed_entity", [])
    routes = sorted({e["route"] for e in entities if e.get("route")})
    stops = sorted({e["stop"] for e in entities if e.get("stop")})
    effect = attrs.get("effect") or "UNKNOWN"
    return {
        "id": alert.get("id"),
        "effect": effect,
        "severity": attrs.get("severity", 0),
        "header": attrs.get("header", ""),
        "short_header": attrs.get("short_header", ""),
        "description": attrs.get("description") or "",
        "service_effect": attrs.get("service_effect") or "",
        "cause": attrs.get("cause") or "",
        "lifecycle": attrs.get("lifecycle") or "",
        "timeframe": attrs.get("timeframe") or "",
        "active_period": attrs.get("active_period") or [],
        "active_now": alert_is_active(attrs, now),
        "updated_at": attrs.get("updated_at"),
        "created_at": attrs.get("created_at"),
        "url": attrs.get("url") or "",
        "affected_routes": routes,
        "affected_stops": stops,
        "route_types": alert_route_types(attrs),
        "modes": alert_modes(attrs),
        "category": "accessibility" if effect in ACCESSIBILITY_EFFECTS else "service",
    }


def alert_priority(alert):
    """Sort key: active service alerts first, then by severity, then most recently updated."""
    updated = parse_time(alert.get("updated_at"))
    updated_ts = updated.timestamp() if updated else 0
    return (
        0 if alert.get("active_now") else 1,
        0 if alert.get("category") == "service" else 1,
        -(alert.get("severity") or 0),
        -updated_ts,
    )


# MBTA severity runs 0-10; 1-2 are informational notices (e.g. "premium parking is
# available at Braintree") that shouldn't mark a whole line as having a service alert.
STATUS_MIN_SEVERITY = 3


def line_status(alerts_for_line):
    """Summarize a line from its *distinct, currently active service* alerts."""
    active = [a for a in alerts_for_line
              if a.get("active_now") and a.get("category") == "service"
              and (a.get("severity") or 0) >= STATUS_MIN_SEVERITY]
    if any((a.get("severity") or 0) >= 7 for a in active):
        status = "alert"
    elif active:
        status = "warning"
    else:
        status = "normal"
    return status, active
