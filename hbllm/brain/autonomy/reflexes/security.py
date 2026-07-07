"""Security Reflexes — threat detection and access control rules.

4 reflexes for security monitoring:
    1. unusual_login_attempt   — failed auth from new device/location
    2. sensitive_action_hours  — dangerous commands outside normal hours
    3. new_device_connected    — unknown device on network/tenant
    4. lock_auto_timeout       — door unlocked too long after last activity

All reflexes are deterministic (Tier 1) — zero LLM cost.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable

from hbllm.brain.autonomy.attention import AttentionEvent
from hbllm.brain.autonomy.reflexes import make_push_message
from hbllm.network.messages import Message

logger = logging.getLogger(__name__)

ReflexRule = Callable[[AttentionEvent], Message | None]


def _unusual_login_attempt(event: AttentionEvent) -> Message | None:
    """Failed authentication or login from unknown device."""
    if event.source.startswith("security.") or event.source.startswith("auth."):
        auth_failed = event.payload.get("auth_failed", False)
        new_device = event.payload.get("new_device", False)
        if auth_failed or new_device:
            device_id = event.payload.get("device_id", "unknown")
            ip_addr = event.payload.get("ip_address", "")
            reason = "Failed authentication" if auth_failed else "New device login"
            return make_push_message(
                title="🔐 Security Alert",
                body=f"{reason} from device {device_id}"
                + (f" (IP: {ip_addr})" if ip_addr else "")
                + ". Was this you?",
                priority="critical" if auth_failed else "high",
            )
    return None


def _sensitive_action_hours(event: AttentionEvent) -> Message | None:
    """Dangerous commands issued outside normal hours (22:00-07:00)."""
    if event.source.startswith("action.") or event.source.startswith("tool."):
        action = event.payload.get("action", "")
        risk_tier = event.payload.get("risk_tier", 0)

        # Only flag Tier 2+ actions outside normal hours
        if risk_tier >= 2:
            hour = datetime.datetime.now().hour
            if hour >= 22 or hour < 7:
                return make_push_message(
                    title="⚠️ After-Hours Action",
                    body=f"High-risk action '{action}' requested at unusual hour ({hour:02d}:00). "
                    f"Requiring confirmation.",
                    priority="high",
                )
    return None


def _new_device_connected(event: AttentionEvent) -> Message | None:
    """Unknown device connected to tenant or network."""
    if event.source.startswith("device.") or event.source.startswith("iot."):
        new_device = event.payload.get("new_device", False)
        device_type = event.payload.get("device_type", "unknown")
        device_name = event.payload.get("device_name") or event.payload.get("device_id", "Unknown")

        if new_device:
            return make_push_message(
                title="📱 New Device",
                body=f"New {device_type} device '{device_name}' connected. Authorize this device?",
                priority="high",
            )
    return None


def _lock_auto_timeout(event: AttentionEvent) -> Message | None:
    """Door unlocked for too long after last activity."""
    if event.source.startswith("iot."):
        device_type = event.payload.get("device_type", "")
        if device_type == "lock" or event.payload.get("lock_state") == "unlocked":
            unlocked_duration_s = event.payload.get("unlocked_duration_s", 0)
            no_activity_s = event.payload.get("no_activity_duration_s", 0)

            # Unlocked > 10 minutes with no activity > 5 minutes
            if unlocked_duration_s > 600 and no_activity_s > 300:
                door_name = event.payload.get("device_name", "Door")
                mins = unlocked_duration_s // 60
                return make_push_message(
                    title="🔓 Lock Timeout",
                    body=f"{door_name} has been unlocked for {mins} minutes with no activity. "
                    f"Lock it?",
                    priority="high",
                )
    return None


def get_security_reflexes() -> dict[str, ReflexRule]:
    """Return all security reflexes."""
    return {
        "unusual_login_attempt": _unusual_login_attempt,
        "sensitive_action_hours": _sensitive_action_hours,
        "new_device_connected": _new_device_connected,
        "lock_auto_timeout": _lock_auto_timeout,
    }
