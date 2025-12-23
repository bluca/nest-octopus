# SPDX-License-Identifier: MPL-2.0
"""
Unit tests for the ntfy notification client.

All tests are mocked to avoid actual network calls.
"""

import io
from datetime import datetime, timedelta, timezone
from http.client import HTTPResponse
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from nest_octopus.ntfy import NtfyClient, Priority


class TestNtfyClientInit:
    """Tests for NtfyClient initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        client = NtfyClient("my-topic")
        assert client.topic == "my-topic"
        assert client.server == "https://ntfy.sh"
        assert client.token is None
        assert client.timeout == 30
        assert client._url == "https://ntfy.sh/my-topic"

    def test_init_with_custom_server(self) -> None:
        """Test initialization with a custom server."""
        client = NtfyClient("my-topic", server="https://ntfy.example.com")
        assert client.server == "https://ntfy.example.com"
        assert client._url == "https://ntfy.example.com/my-topic"

    def test_init_server_trailing_slash_stripped(self) -> None:
        """Test that trailing slash is stripped from server URL."""
        client = NtfyClient("my-topic", server="https://ntfy.example.com/")
        assert client.server == "https://ntfy.example.com"
        assert client._url == "https://ntfy.example.com/my-topic"

    def test_init_with_token(self) -> None:
        """Test initialization with an access token."""
        client = NtfyClient("my-topic", token="tk_secret123")
        assert client.token == "tk_secret123"

    def test_init_with_custom_timeout(self) -> None:
        """Test initialization with a custom timeout."""
        client = NtfyClient("my-topic", timeout=60)
        assert client.timeout == 60

    def test_init_empty_topic_raises(self) -> None:
        """Test that empty topic raises ValueError."""
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            NtfyClient("")

    def test_repr(self) -> None:
        """Test string representation."""
        client = NtfyClient("my-topic", server="https://ntfy.example.com")
        assert repr(client) == "NtfyClient(topic='my-topic', server='https://ntfy.example.com')"


class TestNtfyClientSend:
    """Tests for NtfyClient.send() method."""

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_simple_message(self, mock_urlopen: MagicMock) -> None:
        """Test sending a simple message."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic")
        result = client.send("Hello, world!")

        assert result is True
        mock_urlopen.assert_called_once()

        # Verify the request
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.full_url == "https://ntfy.sh/test-topic"
        assert request.data == b"Hello, world!"
        assert request.method == "POST"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_title(self, mock_urlopen: MagicMock) -> None:
        """Test sending a message with a title."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic")
        result = client.send("Message body", title="My Title")

        assert result is True
        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Title") == "My Title"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_priority(self, mock_urlopen: MagicMock) -> None:
        """Test sending a message with priority."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic")
        result = client.send("Urgent!", priority=Priority.MAX)

        assert result is True
        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Priority") == "5"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_tags(self, mock_urlopen: MagicMock) -> None:
        """Test sending a message with tags."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic")
        result = client.send("Tagged!", tags=["warning", "skull"])

        assert result is True
        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Tags") == "warning,skull"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_click_action(self, mock_urlopen: MagicMock) -> None:
        """Test sending a message with click action."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic")
        result = client.send("Click me!", click="https://example.com")

        assert result is True
        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Click") == "https://example.com"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_markdown(self, mock_urlopen: MagicMock) -> None:
        """Test sending a message with markdown enabled."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic")
        result = client.send("**Bold** text", markdown=True)

        assert result is True
        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Markdown") == "yes"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_auth_token(self, mock_urlopen: MagicMock) -> None:
        """Test sending a message with authentication token."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic", token="tk_secret123")
        result = client.send("Authenticated message")

        assert result is True
        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Authorization") == "Bearer tk_secret123"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_all_options(self, mock_urlopen: MagicMock) -> None:
        """Test sending a message with all options."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic", token="tk_secret")
        result = client.send(
            "Full message",
            title="Full Title",
            priority=Priority.HIGH,
            tags=["tag1", "tag2"],
            click="https://example.com",
            markdown=True,
        )

        assert result is True
        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Title") == "Full Title"
        assert request.get_header("Priority") == "4"
        assert request.get_header("Tags") == "tag1,tag2"
        assert request.get_header("Click") == "https://example.com"
        assert request.get_header("Markdown") == "yes"
        assert request.get_header("Authorization") == "Bearer tk_secret"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_uses_timeout(self, mock_urlopen: MagicMock) -> None:
        """Test that send uses the configured timeout."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic", timeout=45)
        client.send("Test message")

        call_args = mock_urlopen.call_args
        assert call_args[1]["timeout"] == 45


class TestNtfyClientSendErrors:
    """Tests for error handling in NtfyClient.send()."""

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_http_error(self, mock_urlopen: MagicMock) -> None:
        """Test handling of HTTP errors."""
        mock_urlopen.side_effect = HTTPError(
            url="https://ntfy.sh/test",
            code=403,
            msg="Forbidden",
            hdrs=MagicMock(),
            fp=None,
        )

        client = NtfyClient("test-topic")
        result = client.send("Test message")

        assert result is False

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_url_error(self, mock_urlopen: MagicMock) -> None:
        """Test handling of URL errors (network issues)."""
        mock_urlopen.side_effect = URLError("Connection refused")

        client = NtfyClient("test-topic")
        result = client.send("Test message")

        assert result is False

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_unexpected_status(self, mock_urlopen: MagicMock) -> None:
        """Test handling of unexpected status codes."""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = NtfyClient("test-topic")
        result = client.send("Test message")

        assert result is False

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_generic_exception(self, mock_urlopen: MagicMock) -> None:
        """Test handling of generic exceptions."""
        mock_urlopen.side_effect = Exception("Something went wrong")

        client = NtfyClient("test-topic")
        result = client.send("Test message")

        assert result is False


class TestNtfyClientDelay:
    """Tests for the delay/scheduled delivery functionality."""

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_delay_string_duration(self, mock_urlopen: MagicMock) -> None:
        """Test sending with delay specified as duration string."""
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.read.return_value = b'{"id":"test"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = NtfyClient("test-topic")
        client.send("Delayed message", delay="30m")

        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Delay") == "30m"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_delay_string_natural_language(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Test sending with delay specified as natural language."""
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.read.return_value = b'{"id":"test"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = NtfyClient("test-topic")
        client.send("Delayed message", delay="tomorrow 10am")

        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Delay") == "tomorrow 10am"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_delay_unix_timestamp(self, mock_urlopen: MagicMock) -> None:
        """Test sending with delay specified as Unix timestamp integer."""
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.read.return_value = b'{"id":"test"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        timestamp = 1700000000
        client = NtfyClient("test-topic")
        client.send("Delayed message", delay=timestamp)

        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Delay") == "1700000000"

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_delay_datetime(self, mock_urlopen: MagicMock) -> None:
        """Test sending with delay specified as datetime object."""
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.read.return_value = b'{"id":"test"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Use a datetime with timezone info
        target_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        expected_timestamp = str(int(target_time.timestamp()))

        client = NtfyClient("test-topic")
        client.send("Delayed message", delay=target_time)

        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Delay") == expected_timestamp

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_delay_datetime_naive(self, mock_urlopen: MagicMock) -> None:
        """Test sending with delay specified as naive datetime (no tzinfo)."""
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.read.return_value = b'{"id":"test"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Use a naive datetime (no timezone)
        target_time = datetime(2024, 1, 15, 10, 30, 0)

        client = NtfyClient("test-topic")
        client.send("Delayed message", delay=target_time)

        # Should still set the Delay header (converted to local timezone then UTC)
        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Delay") is not None
        # Verify it's a numeric timestamp string
        assert request.get_header("Delay").isdigit()

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_with_delay_timedelta(self, mock_urlopen: MagicMock) -> None:
        """Test sending with delay specified as timedelta."""
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.read.return_value = b'{"id":"test"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = NtfyClient("test-topic")
        client.send("Delayed message", delay=timedelta(hours=2))

        request = mock_urlopen.call_args[0][0]
        delay_header = request.get_header("Delay")
        assert delay_header is not None
        # Should be a Unix timestamp string
        assert delay_header.isdigit()
        # The timestamp should be approximately 2 hours in the future
        delay_timestamp = int(delay_header)
        now_timestamp = int(datetime.now(timezone.utc).timestamp())
        # Allow 5 seconds of tolerance
        expected_diff = 2 * 3600
        assert abs((delay_timestamp - now_timestamp) - expected_diff) < 5

    @patch("nest_octopus.ntfy.urlopen")
    def test_send_without_delay(self, mock_urlopen: MagicMock) -> None:
        """Test that no Delay header is set when delay is not specified."""
        mock_response = MagicMock(spec=HTTPResponse)
        mock_response.read.return_value = b'{"id":"test"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = NtfyClient("test-topic")
        client.send("Immediate message")

        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Delay") is None

    def test_format_delay_string(self) -> None:
        """Test _format_delay with string input."""
        assert NtfyClient._format_delay("30m") == "30m"
        assert NtfyClient._format_delay("2 hours") == "2 hours"
        assert NtfyClient._format_delay("tomorrow 9am") == "tomorrow 9am"

    def test_format_delay_int(self) -> None:
        """Test _format_delay with integer (Unix timestamp) input."""
        assert NtfyClient._format_delay(1700000000) == "1700000000"
        assert NtfyClient._format_delay(0) == "0"

    def test_format_delay_datetime(self) -> None:
        """Test _format_delay with datetime input."""
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        expected = str(int(dt.timestamp()))
        assert NtfyClient._format_delay(dt) == expected

    def test_format_delay_timedelta(self) -> None:
        """Test _format_delay with timedelta input."""
        before = datetime.now(timezone.utc)
        result = NtfyClient._format_delay(timedelta(minutes=30))
        after = datetime.now(timezone.utc)

        result_timestamp = int(result)
        before_expected = int((before + timedelta(minutes=30)).timestamp())
        after_expected = int((after + timedelta(minutes=30)).timestamp())

        assert before_expected <= result_timestamp <= after_expected


class TestPriority:
    """Tests for the Priority enum."""

    def test_priority_values(self) -> None:
        """Test that priority values are correct."""
        assert int(Priority.MIN) == 1
        assert int(Priority.LOW) == 2
        assert int(Priority.DEFAULT) == 3
        assert int(Priority.HIGH) == 4
        assert int(Priority.MAX) == 5

    def test_priority_int_conversion(self) -> None:
        """Test that priorities can be converted to int."""
        assert int(Priority.MIN) == 1
        assert int(Priority.MAX) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
