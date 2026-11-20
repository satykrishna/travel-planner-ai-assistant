"""Atomic shared quotas for Vercel; in-memory limits are development-only."""

import logging
import os
import threading
import time
import uuid
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)
MAX_REQUEST_SECONDS = 180
MINUTE_LIMIT = 3
DAILY_LIMIT = 30
CONCURRENT_LIMIT = 2
LEASE_SECONDS = 240
_NAMESPACE = "{travel-planner}:limits:v1"
_RESERVE = """
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', ARGV[1])
if tonumber(redis.call('GET', KEYS[1]) or '0') >= tonumber(ARGV[4]) then return 1 end
if tonumber(redis.call('GET', KEYS[2]) or '0') >= tonumber(ARGV[5]) then return 2 end
if redis.call('ZCARD', KEYS[3]) >= tonumber(ARGV[6]) then return 3 end
redis.call('INCR', KEYS[1]); redis.call('EXPIRE', KEYS[1], 120)
redis.call('INCR', KEYS[2]); redis.call('EXPIRE', KEYS[2], 172800)
redis.call('ZADD', KEYS[3], ARGV[2], ARGV[3]); redis.call('EXPIRE', KEYS[3], 300)
return 0
"""
_lock = threading.Lock()
_counts = {}
_leases = {}


def redis_credentials():
    url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip().rstrip("/")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    # Fixed provider hosts only; never forward a bearer secret to arbitrary URLs.
    valid = (parsed.scheme == "https" and parsed.hostname and
             parsed.hostname.endswith(".upstash.io") and not parsed.username and
             not parsed.password and port in (None, 443) and
             not parsed.path and not parsed.query and not parsed.fragment)
    return (url, token) if valid and token else None


def quota_ready():
    return not os.getenv("VERCEL") or redis_credentials() is not None


async def redis_command(command):
    credentials = redis_credentials()
    if not credentials:
        raise HTTPException(503, "Live planning is paused while usage protection is configured.")
    url, token = credentials
    try:
        async with httpx.AsyncClient(timeout=5, follow_redirects=False, trust_env=False) as client:
            response = await client.post(url, headers={"Authorization": "Bearer " + token}, json=command)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload or "result" not in payload:
                raise ValueError("Invalid quota response")
            return payload["result"]
    except (httpx.HTTPError, ValueError, TypeError):
        # No in-memory fallback in production: a store outage must not allow spending.
        raise HTTPException(503, "Live planning is temporarily paused. Please try again later.") from None


def keys_at(now):
    return [f"{_NAMESPACE}:minute:{int(now // 60)}", f"{_NAMESPACE}:day:{int(now // 86400)}", f"{_NAMESPACE}:active"]


async def reserve():
    now = time.time()
    keys = keys_at(now)
    lease = uuid.uuid4().hex
    if redis_credentials() or os.getenv("VERCEL"):
        result = await redis_command(["EVAL", _RESERVE, 3, *keys, now, now + LEASE_SECONDS,
                                      lease, MINUTE_LIMIT, DAILY_LIMIT, CONCURRENT_LIMIT])
        if type(result) is not int or result not in (0, 1, 2, 3):
            raise HTTPException(503, "Usage protection is temporarily unavailable.")
    else:
        with _lock:
            for key, expiry in list(_leases.items()):
                if expiry <= now:
                    del _leases[key]
            for key, (_, expiry) in list(_counts.items()):
                if expiry <= now:
                    del _counts[key]
            minute = _counts.get(keys[0], (0, 0))[0]
            day = _counts.get(keys[1], (0, 0))[0]
            result = 1 if minute >= MINUTE_LIMIT else 2 if day >= DAILY_LIMIT else 3 if len(_leases) >= CONCURRENT_LIMIT else 0
            if result == 0:
                _counts[keys[0]] = (minute + 1, now + 120)
                _counts[keys[1]] = (day + 1, now + 172800)
                _leases[lease] = now + LEASE_SECONDS
    if result:
        messages = {1: "Planning is busy. Please wait a minute before trying again.",
                    2: "Today's planning allowance has been reached. Please try again tomorrow.",
                    3: "Two trips are already being researched. Please try again shortly."}
        retry = 60 if result == 1 else max(1, int(86400 - now % 86400)) if result == 2 else 30
        raise HTTPException(429, messages[result], headers={"Retry-After": str(retry)})
    return lease


async def release(lease):
    if redis_credentials() or os.getenv("VERCEL"):
        try:
            await redis_command(["ZREM", f"{_NAMESPACE}:active", lease])
        except HTTPException:
            logger.warning("Usage lease release unavailable; lease will expire")
    else:
        with _lock:
            _leases.pop(lease, None)
