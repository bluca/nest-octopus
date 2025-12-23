# SPDX-License-Identifier: MPL-2.0
"""
ntfy notification client.

Provides a simple interface to send notifications via the ntfy service.
See https://docs.ntfy.sh/publish/ for the full API documentation.
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import List, Optional, Union
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Message priority levels for ntfy notifications."""
    MIN = 1
    LOW = 2
    DEFAULT = 3
    HIGH = 4
    MAX = 5


class NtfyClient:
    """
    Client for sending notifications via ntfy.

    ntfy is a simple HTTP-based pub-sub notification service.
    This client sends notifications to a specific topic.

    Args:
        topic: The topic to publish messages to
        server: The ntfy server URL (default: https://ntfy.sh)
        token: Optional access token for authentication
        timeout: Request timeout in seconds (default: 30)

    Example:
        >>> client = NtfyClient("my-alerts")
        >>> client.send("Backup completed successfully")
        >>> client.send("Disk space low", title="Warning", priority=Priority.HIGH)
    """

    def __init__(
        self,
        topic: str,
        server: str = "https://ntfy.sh",
        token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """
        Initialize the ntfy client.

        Args:
            topic: The topic to publish messages to
            server: The ntfy server URL (default: https://ntfy.sh)
            token: Optional access token for authentication
            timeout: Request timeout in seconds (default: 30)
        """
        if not topic:
            raise ValueError("Topic cannot be empty")

        self.topic = topic
        self.server = server.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._url = urljoin(self.server + "/", self.topic)

    def send(
        self,
        message: str,
        title: Optional[str] = None,
        priority: Optional[Priority] = None,
        tags: Optional[List[str]] = None,
        click: Optional[str] = None,
        markdown: bool = False,
        delay: Optional[Union[str, int, datetime, timedelta]] = None,
    ) -> bool:
        """
        Send a notification to the configured topic.

        Args:
            message: The notification message body
            title: Optional message title
            priority: Optional message priority (1-5, use Priority enum)
            tags: Optional list of tags (can include emoji shortcodes)
            click: Optional URL to open when notification is clicked
            markdown: Whether to enable markdown formatting
            delay: Optional scheduled delivery time. Can be:
                   - str: Duration ("30m", "2h", "1 day") or time ("10am", "tomorrow 3pm")
                   - int: Unix timestamp
                   - datetime: Absolute time (will be converted to Unix timestamp)
                   - timedelta: Delay from now (will be converted to Unix timestamp)
                   Note: Minimum delay is 10 seconds, maximum is 3 days.

        Returns:
            True if the notification was sent successfully, False otherwise
        """
        headers: dict[str, str] = {}

        if title:
            headers["Title"] = title

        if priority is not None:
            headers["Priority"] = str(int(priority))

        if tags:
            headers["Tags"] = ",".join(tags)

        if click:
            headers["Click"] = click

        if markdown:
            headers["Markdown"] = "yes"

        if delay is not None:
            headers["Delay"] = self._format_delay(delay)

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            request = Request(
                self._url,
                data=message.encode("utf-8"),
                headers=headers,
                method="POST",
            )

            with urlopen(request, timeout=self.timeout) as response:
                status = response.status
                if status == 200:
                    logger.debug(f"Notification sent to {self.topic}: {message[:50]}...")
                    return True
                else:
                    logger.warning(f"Unexpected status {status} from ntfy server")
                    return False

        except HTTPError as e:
            logger.error(f"HTTP error sending notification: {e.code} {e.reason}")
            return False
        except URLError as e:
            logger.error(f"URL error sending notification: {e.reason}")
            return False
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False

    @staticmethod
    def _format_delay(delay: Union[str, int, datetime, timedelta]) -> str:
        """
        Format a delay value for the ntfy Delay header.

        Args:
            delay: The delay value in various formats

        Returns:
            Formatted string for the Delay header
        """
        if isinstance(delay, str):
            # Pass through string values (e.g., "30m", "tomorrow 10am")
            return delay
        elif isinstance(delay, int):
            # Unix timestamp
            return str(delay)
        elif isinstance(delay, datetime):
            # Convert datetime to Unix timestamp
            if delay.tzinfo is None:
                # Assume local time, convert to UTC
                delay = delay.astimezone(timezone.utc)
            return str(int(delay.timestamp()))
        elif isinstance(delay, timedelta):
            # Convert timedelta to Unix timestamp (now + delta)
            target = datetime.now(timezone.utc) + delay
            return str(int(target.timestamp()))
        else:
            raise TypeError(
                f"delay must be str, int, datetime, or timedelta, not {type(delay).__name__}"
            )

    def __repr__(self) -> str:
        """Return string representation of the client."""
        return f"NtfyClient(topic={self.topic!r}, server={self.server!r})"
