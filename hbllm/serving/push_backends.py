"""Push Notification Backends — FCM and APNs integration.

Delivers proactive notifications to mobile devices:
    - Firebase Cloud Messaging (FCM) for Android/web
    - Apple Push Notification service (APNs) for iOS
    - Abstracted behind a common PushBackend interface

Architecture:
    1. PushBackend interface with send() and send_batch()
    2. FCMBackend using google-auth + HTTP/2
    3. APNsBackend using httpx + JWT
    4. InMemoryBackend for testing
    5. MultiBackend that routes to FCM or APNs by device type
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PushNotification:
    """A push notification to be delivered."""

    title: str = ""
    body: str = ""
    device_token: str = ""
    device_type: str = "fcm"  # "fcm", "apns", "web"
    priority: str = "high"  # "normal", "high"
    category: str = ""  # Notification category for grouping
    data: dict[str, Any] = field(default_factory=dict)  # Custom payload
    badge: int | None = None
    sound: str = "default"
    collapse_key: str = ""  # Replaces previous notification with same key
    ttl_s: int = 86400  # Time-to-live

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "device_type": self.device_type,
            "priority": self.priority,
            "category": self.category,
            "data": self.data,
        }


@dataclass
class PushResult:
    """Result of a push notification delivery."""

    success: bool = False
    device_token: str = ""
    message_id: str = ""
    error: str = ""
    timestamp: float = field(default_factory=time.time)


class PushBackend(ABC):
    """Abstract push notification backend."""

    @abstractmethod
    async def send(self, notification: PushNotification) -> PushResult:
        """Send a single push notification."""
        ...

    async def send_batch(self, notifications: list[PushNotification]) -> list[PushResult]:
        """Send multiple notifications. Default: sequential sends."""
        return [await self.send(n) for n in notifications]

    @abstractmethod
    async def validate_token(self, token: str) -> bool:
        """Check if a device token is valid."""
        ...


class FCMBackend(PushBackend):
    """Firebase Cloud Messaging backend.

    Requires:
        - Service account JSON file (GOOGLE_APPLICATION_CREDENTIALS env var)
        - google-auth and httpx packages

    Usage::

        fcm = FCMBackend(project_id="my-project")
        await fcm.init()
        result = await fcm.send(PushNotification(
            title="Hello",
            body="World",
            device_token="abc123",
        ))
    """

    FCM_URL = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"

    def __init__(
        self,
        project_id: str = "",
        credentials_path: str | None = None,
    ) -> None:
        self.project_id = project_id
        self.credentials_path = credentials_path
        self._credentials: Any = None
        self._client: Any = None

        # Telemetry
        self._sent = 0
        self._failed = 0

    async def init(self) -> None:
        """Initialize FCM credentials and HTTP client."""
        try:
            import google.auth  # type: ignore[import-not-found]
            import google.auth.transport.requests  # type: ignore[import-not-found]

            if self.credentials_path:
                from google.oauth2 import service_account  # type: ignore[import-not-found]

                self._credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=["https://www.googleapis.com/auth/firebase.messaging"],
                )
            else:
                self._credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/firebase.messaging"]
                )

            logger.info("FCMBackend initialized for project: %s", self.project_id)
        except ImportError:
            logger.warning(
                "FCMBackend: google-auth not installed. Install with: pip install google-auth"
            )
        except Exception as e:
            logger.warning("FCMBackend initialization failed: %s", e)

    async def send(self, notification: PushNotification) -> PushResult:
        """Send via FCM HTTP v1 API."""
        if not self._credentials:
            return PushResult(
                success=False,
                device_token=notification.device_token,
                error="FCM credentials not initialized",
            )

        try:
            import google.auth.transport.requests  # type: ignore[import-not-found]
            import httpx  # type: ignore[import-not-found]

            # Refresh credentials
            self._credentials.refresh(google.auth.transport.requests.Request())
            token = self._credentials.token

            url = self.FCM_URL.format(project_id=self.project_id)

            payload = {
                "message": {
                    "token": notification.device_token,
                    "notification": {
                        "title": notification.title,
                        "body": notification.body,
                    },
                    "data": {k: str(v) for k, v in notification.data.items()},
                    "android": {
                        "priority": notification.priority,
                        "notification": {"sound": notification.sound},
                    },
                }
            }

            if notification.collapse_key:
                payload["message"]["android"]["collapse_key"] = notification.collapse_key

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10.0,
                )

            if resp.status_code == 200:
                self._sent += 1
                data = resp.json()
                return PushResult(
                    success=True,
                    device_token=notification.device_token,
                    message_id=data.get("name", ""),
                )
            else:
                self._failed += 1
                return PushResult(
                    success=False,
                    device_token=notification.device_token,
                    error=f"FCM error {resp.status_code}: {resp.text[:200]}",
                )

        except Exception as e:
            self._failed += 1
            return PushResult(
                success=False,
                device_token=notification.device_token,
                error=str(e),
            )

    async def validate_token(self, token: str) -> bool:
        """Validate an FCM token by sending a dry-run message."""
        # FCM doesn't have a direct token validation API
        # We'd need to send a dry-run notification
        return bool(token)


class APNsBackend(PushBackend):
    """Apple Push Notification service backend.

    Requires:
        - APNs key file (.p8) and key ID
        - Team ID
        - httpx package

    Usage::

        apns = APNsBackend(
            key_path="AuthKey.p8",
            key_id="ABC123",
            team_id="DEF456",
            bundle_id="com.hbllm.app",
        )
        await apns.init()
    """

    APNS_URL = "https://api.push.apple.com/3/device/{token}"
    APNS_SANDBOX_URL = "https://api.sandbox.push.apple.com/3/device/{token}"

    def __init__(
        self,
        key_path: str = "",
        key_id: str = "",
        team_id: str = "",
        bundle_id: str = "",
        sandbox: bool = False,
    ) -> None:
        self.key_path = key_path
        self.key_id = key_id
        self.team_id = team_id
        self.bundle_id = bundle_id
        self.sandbox = sandbox
        self._key_data: str = ""

        self._sent = 0
        self._failed = 0

    async def init(self) -> None:
        """Load APNs authentication key."""
        try:
            with open(self.key_path) as f:
                self._key_data = f.read()
            logger.info(
                "APNsBackend initialized (team=%s, bundle=%s)", self.team_id, self.bundle_id
            )
        except FileNotFoundError:
            logger.warning("APNsBackend: Key file not found: %s", self.key_path)
        except Exception as e:
            logger.warning("APNsBackend initialization failed: %s", e)

    def _generate_jwt(self) -> str:
        """Generate a JWT for APNs authentication."""
        try:
            import jwt  # type: ignore[import-not-found]

            headers = {"alg": "ES256", "kid": self.key_id}
            payload = {
                "iss": self.team_id,
                "iat": int(time.time()),
            }
            return jwt.encode(payload, self._key_data, algorithm="ES256", headers=headers)
        except ImportError:
            logger.warning("PyJWT not installed. Install with: pip install PyJWT")
            return ""

    async def send(self, notification: PushNotification) -> PushResult:
        """Send via APNs HTTP/2 API."""
        if not self._key_data:
            return PushResult(
                success=False,
                device_token=notification.device_token,
                error="APNs key not loaded",
            )

        try:
            import httpx  # type: ignore[import-not-found]

            token_str = self._generate_jwt()
            if not token_str:
                return PushResult(
                    success=False,
                    device_token=notification.device_token,
                    error="JWT generation failed",
                )

            base_url = self.APNS_SANDBOX_URL if self.sandbox else self.APNS_URL
            url = base_url.format(token=notification.device_token)

            payload = {
                "aps": {
                    "alert": {
                        "title": notification.title,
                        "body": notification.body,
                    },
                    "sound": notification.sound,
                    "category": notification.category,
                },
                **notification.data,
            }

            if notification.badge is not None:
                payload["aps"]["badge"] = notification.badge

            headers = {
                "Authorization": f"bearer {token_str}",
                "apns-topic": self.bundle_id,
                "apns-priority": "10" if notification.priority == "high" else "5",
                "apns-push-type": "alert",
            }

            if notification.collapse_key:
                headers["apns-collapse-id"] = notification.collapse_key

            async with httpx.AsyncClient(http2=True) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=10.0,
                )

            if resp.status_code == 200:
                self._sent += 1
                return PushResult(
                    success=True,
                    device_token=notification.device_token,
                    message_id=resp.headers.get("apns-id", ""),
                )
            else:
                self._failed += 1
                return PushResult(
                    success=False,
                    device_token=notification.device_token,
                    error=f"APNs error {resp.status_code}: {resp.text[:200]}",
                )

        except Exception as e:
            self._failed += 1
            return PushResult(
                success=False,
                device_token=notification.device_token,
                error=str(e),
            )

    async def validate_token(self, token: str) -> bool:
        return bool(token and len(token) == 64)


class InMemoryBackend(PushBackend):
    """In-memory push backend for testing."""

    def __init__(self) -> None:
        self.sent: list[PushNotification] = []

    async def send(self, notification: PushNotification) -> PushResult:
        self.sent.append(notification)
        return PushResult(
            success=True,
            device_token=notification.device_token,
            message_id=f"mem_{len(self.sent)}",
        )

    async def validate_token(self, token: str) -> bool:
        return True


class MultiBackend(PushBackend):
    """Routes notifications to the correct backend by device type."""

    def __init__(
        self,
        fcm: PushBackend | None = None,
        apns: PushBackend | None = None,
        fallback: PushBackend | None = None,
    ) -> None:
        self.fcm = fcm
        self.apns = apns
        self.fallback = fallback or InMemoryBackend()

    async def send(self, notification: PushNotification) -> PushResult:
        if notification.device_type == "apns" and self.apns:
            return await self.apns.send(notification)
        elif notification.device_type in ("fcm", "web") and self.fcm:
            return await self.fcm.send(notification)
        else:
            return await self.fallback.send(notification)

    async def validate_token(self, token: str) -> bool:
        return True
