"""Best-effort rate limits within one worker; not a distributed spending cap."""

import threading
import time
import uuid

from fastapi import HTTPException

MAX_REQUEST_SECONDS = 180
MINUTE_LIMIT = 3
DAILY_LIMIT = 30
CONCURRENT_LIMIT = 2
LEASE_SECONDS = 240
_lock = threading.Lock()
_counts = {}
_leases = {}


async def reserve():
    now = time.time()
    minute_key = ("minute", int(now // 60))
    day_key = ("day", int(now // 86400))
    lease = uuid.uuid4().hex
    with _lock:
        for key, expiry in list(_leases.items()):
            if expiry <= now:
                del _leases[key]
        for key, (_, expiry) in list(_counts.items()):
            if expiry <= now:
                del _counts[key]
        minute = _counts.get(minute_key, (0, 0))[0]
        day = _counts.get(day_key, (0, 0))[0]
        result = 1 if minute >= MINUTE_LIMIT else 2 if day >= DAILY_LIMIT else 3 if len(_leases) >= CONCURRENT_LIMIT else 0
        if result == 0:
            _counts[minute_key] = (minute + 1, now + 120)
            _counts[day_key] = (day + 1, now + 172800)
            _leases[lease] = now + LEASE_SECONDS
    if result:
        messages = {
            1: "Too many plans were requested in a short time. Please wait one minute before trying again.",
            2: "This planner has reached its daily allowance on this server. Please try again tomorrow or explore the example plan.",
            3: "The planner is researching other trips right now. Please try again in 30 seconds.",
        }
        retry = 60 if result == 1 else max(1, int(86400 - now % 86400)) if result == 2 else 30
        raise HTTPException(429, messages[result], headers={"Retry-After": str(retry)})
    return lease


async def release(lease):
    with _lock:
        _leases.pop(lease, None)


def retry_after_seconds():
    """Earliest retry from current worker quotas, after this request releases its slot."""
    now = time.time()
    with _lock:
        minute = _counts.get(("minute", int(now // 60)), (0, 0))[0]
        day = _counts.get(("day", int(now // 86400)), (0, 0))[0]
    if day >= DAILY_LIMIT:
        return max(1, int(86400 - now % 86400) + 1)
    if minute >= MINUTE_LIMIT:
        return max(1, int(60 - now % 60) + 1)
    return 0
