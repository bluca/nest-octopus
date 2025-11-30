# SPDX-License-Identifier: MPL-2.0
"""
Comprehensive offline tests for Google Nest Thermostat API client.

These tests verify the functionality of the NestThermostatClient class without
making any real network requests. All API responses are mocked using fixtures.
"""

import json
import socket
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from nest_octopus.nest_thermostat import (
    EcoMode,
    FanMode,
    NestAPIError,
    NestThermostatClient,
    ThermostatMode,
    ThermostatStatus,
)


# Fixture directory paths
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "nest"
VALID_DIR = FIXTURES_DIR / "valid"
INVALID_DIR = FIXTURES_DIR / "invalid"


@pytest.fixture(autouse=True)
def block_network_access(monkeypatch):
    """
    Prevent any real network access during tests.

    This fixture is automatically applied to all tests in this module.
    It monkeypatches socket.socket to raise an error if any code tries
    to create a network connection.
    """
    def guard(*args, **kwargs):
        raise RuntimeError(
            "Network access detected! Tests should not make real network calls."
        )

    monkeypatch.setattr(socket, "socket", guard)


@pytest.fixture(autouse=True)
def mock_token_refresh():
    """
    Mock the OAuth2 token refresh for all tests.

    This prevents tests from trying to refresh tokens, since all tests
    should mock the actual API calls instead.
    """
    with patch('nest_octopus.nest_thermostat.requests.post') as mock_post:
        # Use the OAuth2 token refresh response format from Google
        fixture_data = load_fixture("oauth_token_response_full.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        yield mock_post


def load_fixture(filename: str) -> dict:
    """
    Load a JSON fixture file.

    Args:
        filename: Name of the fixture file (with or without .json extension)

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If the fixture file doesn't exist
    """
    if not filename.endswith('.json'):
        filename += '.json'

    # Try valid directory first, then invalid
    valid_path = VALID_DIR / filename
    invalid_path = INVALID_DIR / filename

    if valid_path.exists():
        with open(valid_path) as f:
            return json.load(f)
    elif invalid_path.exists():
        with open(invalid_path) as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Fixture not found: {filename}")


def create_test_client():
    """Create a NestThermostatClient for testing with standard credentials."""
    return NestThermostatClient(
        project_id="test-project",
        refresh_token="test-refresh-token",
        client_id="test-client-id",
        client_secret="test-client-secret"
    )


class TestThermostatStatus:
    """Test the ThermostatStatus dataclass."""

    def test_parse_device_data_heat_mode(self):
        """Test parsing device data in HEAT mode."""
        data = load_fixture("device_heat_mode.json")
        status = ThermostatStatus(data)

        assert status.device_id == "enterprises/project-id/devices/device-id-1"
        assert status.mode == "HEAT"
        assert status.eco_mode == "OFF"
        assert status.hvac_status == "HEATING"
        assert status.heat_setpoint == 20.0
        assert status.cool_setpoint is None
        assert status.temperature == 18.5
        assert status.humidity == 45.0

    def test_parse_device_data_cool_mode(self):
        """Test parsing device data in COOL mode."""
        data = load_fixture("device_cool_mode.json")
        status = ThermostatStatus(data)

        assert status.device_id == "enterprises/project-id/devices/device-id-2"
        assert status.mode == "COOL"
        assert status.eco_mode == "OFF"
        assert status.hvac_status == "COOLING"
        assert status.heat_setpoint is None
        assert status.cool_setpoint == 22.0
        assert status.temperature == 24.5

    def test_parse_device_data_heatcool_mode(self):
        """Test parsing device data in HEATCOOL mode."""
        data = load_fixture("device_heatcool_mode.json")
        status = ThermostatStatus(data)

        assert status.device_id == "enterprises/project-id/devices/device-id-3"
        assert status.mode == "HEATCOOL"
        assert status.eco_mode == "OFF"
        assert status.hvac_status == "OFF"
        assert status.heat_setpoint == 18.0
        assert status.cool_setpoint == 24.0
        assert status.temperature == 21.0

    def test_parse_device_data_eco_mode(self):
        """Test parsing device data in ECO mode."""
        data = load_fixture("device_eco_mode.json")
        status = ThermostatStatus(data)

        assert status.device_id == "enterprises/project-id/devices/device-id-4"
        assert status.eco_mode == "MANUAL_ECO"
        assert status.temperature == 19.0

    def test_parse_device_data_offline(self):
        """Test parsing device data for offline device."""
        data = load_fixture("device_offline.json")
        status = ThermostatStatus(data)

        assert status.device_id == "enterprises/project-id/devices/device-id-5"
        assert status.mode == "OFF"


class TestNestThermostatClientInit:
    """Test NestThermostatClient initialization."""

    def test_init_success(self):
        """Test successful client initialization."""
        client = NestThermostatClient(
            project_id="test-project",
            refresh_token="test-refresh-token",
            client_id="test-client-id",
            client_secret="test-client-secret"
        )

        assert client.project_id == "test-project"
        assert client.refresh_token == "test-refresh-token"
        assert client.client_id == "test-client-id"
        assert client.client_secret == "test-client-secret"
        assert client.base_url == "https://smartdevicemanagement.googleapis.com/v1"
        # Access token comes from the OAuth2 token refresh fixture
        assert client.access_token == "ya29.a0AfB_byD1234567890abcdefghijklmnopqrstuvwxyz"

    def test_context_manager(self):
        """Test using client as context manager."""
        with NestThermostatClient(
            "test-project",
            "test-refresh-token",
            "test-client-id",
            "test-client-secret"
        ) as client:
            assert client.project_id == "test-project"
            # Session exists during context
            assert client.session is not None

    def test_token_refresh_on_init(self, mock_token_refresh):
        """Test that refresh token is used during initialization."""
        client = create_test_client()

        # Verify token refresh was called during init
        assert mock_token_refresh.called
        call_args = mock_token_refresh.call_args

        # Verify correct parameters were sent for refresh_token grant
        assert call_args[1]['params']['grant_type'] == 'refresh_token'
        assert call_args[1]['params']['client_id'] == 'test-client-id'
        assert call_args[1]['params']['client_secret'] == 'test-client-secret'
        assert call_args[1]['params']['refresh_token'] == 'test-refresh-token'

        # Verify access token was set from refresh response
        assert client.access_token == 'ya29.a0AfB_byD1234567890abcdefghijklmnopqrstuvwxyz'
        assert client.token_expiry is not None


class TestOAuth2TokenRefresh:
    """Test OAuth2 token refresh functionality."""

    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_token_refresh_full_response(self, mock_post):
        """Test token refresh with full response."""
        fixture_data = load_fixture("oauth_token_response_full.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()

        # Verify all fields are properly handled
        assert client.access_token == "ya29.a0AfB_byD1234567890abcdefghijklmnopqrstuvwxyz"
        assert client.refresh_token == "test-refresh-token"
        assert client.token_expiry is not None

        # Verify authorization header is set
        assert 'Authorization' in client.session.headers
        assert client.session.headers['Authorization'] == f'Bearer {client.access_token}'

        # Verify the correct grant type was used
        call_args = mock_post.call_args
        assert call_args[1]['params']['grant_type'] == 'refresh_token'
        assert call_args[1]['params']['refresh_token'] == 'test-refresh-token'

    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_token_refresh_minimal_response(self, mock_post):
        """Test token refresh with minimal response."""
        fixture_data = load_fixture("oauth_token_response_minimal.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()

        # Should work with minimal response (access_token + expires_in)
        assert client.access_token == "short-lived-token"
        assert client.token_expiry is not None

    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_token_refresh_on_expiry(self, mock_post):
        """Test that token is refreshed when expired."""
        # First call: initial refresh
        first_fixture = load_fixture("oauth_token_response_full.json")
        mock_response = Mock()
        mock_response.json.return_value = first_fixture
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()
        first_token = client.access_token

        # Verify first call was refresh_token grant
        first_call_args = mock_post.call_args
        assert first_call_args[1]['params']['grant_type'] == 'refresh_token'

        # Reset mock for second call
        mock_post.reset_mock()

        # Set up second token response (refresh)
        second_fixture = load_fixture("oauth_token_response_full.json")
        mock_response.json.return_value = second_fixture
        mock_post.return_value = mock_response

        # Force token to be expired
        from datetime import datetime, timedelta
        client.token_expiry = datetime.now() - timedelta(seconds=1)

        # Call _ensure_valid_token which should trigger refresh
        client._ensure_valid_token()

        # Verify second call was also refresh_token grant
        assert mock_post.called
        second_call_args = mock_post.call_args
        assert second_call_args[1]['params']['grant_type'] == 'refresh_token'
        assert second_call_args[1]['params']['refresh_token'] == 'test-refresh-token'

    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_token_refresh_short_expiry(self, mock_post):
        """Test token with short expiry time."""
        fixture_data = load_fixture("oauth_token_response_short_expiry.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()

        # With short expiry, token should be set but expiry should be soon
        assert client.access_token == "token-with-short-expiry"
        assert client.token_expiry is not None

        # Token expiry should be less than original (due to safety margin)
        from datetime import datetime
        time_until_expiry = (client.token_expiry - datetime.now()).total_seconds()
        # Should be very close to 0 or negative (already expired due to safety margin)
        assert time_until_expiry <= 5  # Allow small margin for test execution time

    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_token_refresh_missing_access_token(self, mock_post):
        """Test error when access_token is missing from response."""
        fixture_data = load_fixture("oauth_missing_access_token.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with pytest.raises(NestAPIError) as exc_info:
            create_test_client()

        assert "access_token" in str(exc_info.value).lower()

    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_token_refresh_http_error(self, mock_post):
        """Test handling of HTTP errors during token refresh."""
        fixture_data = load_fixture("oauth_error_invalid_grant.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.status_code = 400
        mock_response.text = "invalid_grant"

        http_error = requests.exceptions.HTTPError()
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_post.return_value = mock_response

        with pytest.raises(NestAPIError) as exc_info:
            create_test_client()

        assert "400" in str(exc_info.value)


class TestExpiredTokenHandling:
    """Test automatic handling of expired tokens during API calls."""

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_list_devices_with_expired_token(self, mock_post, mock_get):
        """Test that list_devices automatically refreshes expired token and retries."""
        # Setup: Mock initial token refresh
        initial_data = load_fixture("oauth_token_response_full.json")
        initial_response = Mock()
        initial_response.json.return_value = initial_data
        initial_response.raise_for_status.return_value = None

        # Setup: Mock second token refresh
        refresh_data = load_fixture("oauth_token_response_full.json")
        refresh_response = Mock()
        refresh_response.json.return_value = refresh_data
        refresh_response.raise_for_status.return_value = None

        # POST returns initial refresh first, then second refresh
        mock_post.side_effect = [initial_response, refresh_response]

        # First GET call: return expired token error
        expired_error_data = load_fixture("error_expired_token.json")
        expired_response = Mock()
        expired_response.status_code = 401
        expired_response.json.return_value = expired_error_data
        expired_response.text = json.dumps(expired_error_data)

        # Second GET call: return success after token refresh
        success_data = load_fixture("list_devices.json")
        success_response = Mock()
        success_response.json.return_value = success_data
        success_response.raise_for_status.return_value = None

        # Configure mock to return error first, then success
        http_error = requests.exceptions.HTTPError()
        http_error.response = expired_response
        expired_response.raise_for_status.side_effect = http_error

        mock_get.side_effect = [expired_response, success_response]

        client = create_test_client()
        devices = client.list_devices()

        # Verify that we got the devices after retry
        assert len(devices) == 1
        assert devices[0]["name"] == "enterprises/project-id/devices/device-id-1"

        # Verify token refresh was called
        assert mock_post.call_count == 2  # Initial refresh + refresh after 401

        # Verify GET was called twice (initial + retry)
        assert mock_get.call_count == 2

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_get_device_with_expired_token(self, mock_post, mock_get):
        """Test that get_device automatically refreshes expired token and retries."""
        # Setup: Mock initial token refresh
        initial_data = load_fixture("oauth_token_response_full.json")
        initial_response = Mock()
        initial_response.json.return_value = initial_data
        initial_response.raise_for_status.return_value = None

        # Setup: Mock second token refresh
        refresh_data = load_fixture("oauth_token_response_full.json")
        refresh_response = Mock()
        refresh_response.json.return_value = refresh_data
        refresh_response.raise_for_status.return_value = None

        # POST returns initial refresh first, then second refresh
        mock_post.side_effect = [initial_response, refresh_response]

        # First GET call: return expired token error
        expired_error_data = load_fixture("error_expired_token.json")
        expired_response = Mock()
        expired_response.status_code = 401
        expired_response.json.return_value = expired_error_data
        expired_response.text = json.dumps(expired_error_data)

        # Second GET call: return success after token refresh
        success_data = load_fixture("device_heat_mode.json")
        success_response = Mock()
        success_response.json.return_value = success_data
        success_response.raise_for_status.return_value = None

        # Configure mock to return error first, then success
        http_error = requests.exceptions.HTTPError()
        http_error.response = expired_response
        expired_response.raise_for_status.side_effect = http_error

        mock_get.side_effect = [expired_response, success_response]

        client = create_test_client()
        status = client.get_device("enterprises/project-id/devices/device-id-1")

        # Verify that we got the device status after retry
        assert status.device_id == "enterprises/project-id/devices/device-id-1"
        assert status.mode == "HEAT"

        # Verify token refresh was called
        assert mock_post.call_count == 2  # Initial refresh + refresh after 401

        # Verify GET was called twice (initial + retry)
        assert mock_get.call_count == 2

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_mode_with_expired_token(self, mock_session_post):
        """Test that set_mode automatically refreshes expired token and retries."""
        # We need to mock both requests.post (for OAuth) and Session.post (for commands)
        with patch('nest_octopus.nest_thermostat.requests.post') as mock_oauth_post:
            # First: auth code exchange via requests.post
            auth_data = load_fixture("oauth_auth_code_exchange_full.json")
            auth_response = Mock()
            auth_response.json.return_value = auth_data
            auth_response.raise_for_status.return_value = None

            # Second: token refresh via requests.post
            token_data = load_fixture("oauth_token_response_full.json")
            token_response = Mock()
            token_response.json.return_value = token_data
            token_response.raise_for_status.return_value = None

            mock_oauth_post.side_effect = [auth_response, token_response]

            # Third: executeCommand via Session.post - returns expired token error
            expired_error_data = load_fixture("error_expired_token.json")
            expired_response = Mock()
            expired_response.status_code = 401
            expired_response.json.return_value = expired_error_data
            expired_response.text = json.dumps(expired_error_data)

            # Fourth: executeCommand retry via Session.post - returns success
            success_response = Mock()
            success_response.text = ""  # Empty response on success
            success_response.raise_for_status.return_value = None

            # Configure mock to return error first, then success
            http_error = requests.exceptions.HTTPError()
            http_error.response = expired_response
            expired_response.raise_for_status.side_effect = http_error

            mock_session_post.side_effect = [expired_response, success_response]

            client = create_test_client()
            result = client.set_mode("enterprises/project-id/devices/device-id-1", ThermostatMode.COOL)

            # Verify command succeeded after retry
            assert result == {}

            # Verify OAuth POST was called twice: auth code exchange + token refresh
            assert mock_oauth_post.call_count == 2

            # Verify Session POST was called twice: failed command + successful retry
            assert mock_session_post.call_count == 2

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_heat_with_expired_token(self, mock_session_post):
        """Test that set_heat automatically refreshes expired token and retries."""
        with patch('nest_octopus.nest_thermostat.requests.post') as mock_oauth_post:
            # First: auth code exchange
            auth_data = load_fixture("oauth_auth_code_exchange_full.json")
            auth_response = Mock()
            auth_response.json.return_value = auth_data
            auth_response.raise_for_status.return_value = None

            # Second: token refresh
            token_data = load_fixture("oauth_token_response_full.json")
            token_response = Mock()
            token_response.json.return_value = token_data
            token_response.raise_for_status.return_value = None

            mock_oauth_post.side_effect = [auth_response, token_response]

            # Third: executeCommand - expired token error
            expired_error_data = load_fixture("error_expired_token.json")
            expired_response = Mock()
            expired_response.status_code = 401
            expired_response.json.return_value = expired_error_data
            expired_response.text = json.dumps(expired_error_data)

            # Fourth: executeCommand retry - success
            success_response = Mock()
            success_response.text = ""
            success_response.raise_for_status.return_value = None

            http_error = requests.exceptions.HTTPError()
            http_error.response = expired_response
            expired_response.raise_for_status.side_effect = http_error

            mock_session_post.side_effect = [expired_response, success_response]

            client = create_test_client()
            result = client.set_heat("enterprises/project-id/devices/device-id-1", 22.0)

            # Verify command succeeded after retry
            assert result == {}

            # Verify OAuth and Session POST calls
            assert mock_oauth_post.call_count == 2
            assert mock_session_post.call_count == 2

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_fan_with_expired_token(self, mock_session_post):
        """Test that set_fan automatically refreshes expired token and retries."""
        with patch('nest_octopus.nest_thermostat.requests.post') as mock_oauth_post:
            # First: auth code exchange
            auth_data = load_fixture("oauth_auth_code_exchange_full.json")
            auth_response = Mock()
            auth_response.json.return_value = auth_data
            auth_response.raise_for_status.return_value = None

            # Second: token refresh
            token_data = load_fixture("oauth_token_response_full.json")
            token_response = Mock()
            token_response.json.return_value = token_data
            token_response.raise_for_status.return_value = None

            mock_oauth_post.side_effect = [auth_response, token_response]

            # Third: executeCommand - expired token error
            expired_error_data = load_fixture("error_expired_token.json")
            expired_response = Mock()
            expired_response.status_code = 401
            expired_response.json.return_value = expired_error_data
            expired_response.text = json.dumps(expired_error_data)

            # Fourth: executeCommand retry - success
            success_response = Mock()
            success_response.text = ""
            success_response.raise_for_status.return_value = None

            http_error = requests.exceptions.HTTPError()
            http_error.response = expired_response
            expired_response.raise_for_status.side_effect = http_error

            mock_session_post.side_effect = [expired_response, success_response]

            client = create_test_client()
            result = client.set_fan("enterprises/project-id/devices/device-id-1", FanMode.ON, 900)

            # Verify command succeeded after retry
            assert result == {}

            # Verify OAuth and Session POST calls
            assert mock_oauth_post.call_count == 2
            assert mock_session_post.call_count == 2

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_expired_token_only_retries_once(self, mock_post, mock_get):
        """Test that expired token error only retries once, not infinitely."""
        # Setup: Mock initial auth code exchange
        auth_data = load_fixture("oauth_auth_code_exchange_full.json")
        auth_response = Mock()
        auth_response.json.return_value = auth_data
        auth_response.raise_for_status.return_value = None

        # Setup: Mock token refresh
        token_data = load_fixture("oauth_token_response_full.json")
        token_response = Mock()
        token_response.json.return_value = token_data
        token_response.raise_for_status.return_value = None

        # POST returns auth code exchange first, then token refresh
        mock_post.side_effect = [auth_response, token_response]

        # Both GET calls: return expired token error
        expired_error_data = load_fixture("error_expired_token.json")
        expired_response = Mock()
        expired_response.status_code = 401
        expired_response.json.return_value = expired_error_data
        expired_response.text = json.dumps(expired_error_data)

        # Configure mock to return error both times
        http_error = requests.exceptions.HTTPError()
        http_error.response = expired_response
        expired_response.raise_for_status.side_effect = http_error

        mock_get.return_value = expired_response

        client = create_test_client()

        # Should raise NestAPIError after retry also fails
        with pytest.raises(NestAPIError) as exc_info:
            client.list_devices()

        assert "401" in str(exc_info.value)

        # Verify GET was called exactly twice (initial + one retry)
        assert mock_get.call_count == 2

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    @patch('nest_octopus.nest_thermostat.requests.post')
    def test_non_token_401_error_not_retried(self, mock_post, mock_get):
        """Test that 401 errors without UNAUTHENTICATED status are not retried."""
        # Setup: Mock initial auth code exchange
        auth_data = load_fixture("oauth_auth_code_exchange_full.json")
        auth_response = Mock()
        auth_response.json.return_value = auth_data
        auth_response.raise_for_status.return_value = None
        mock_post.return_value = auth_response

        # GET call: return 401 with different error (not UNAUTHENTICATED)
        error_data = {"error": {"code": 401, "message": "Some other auth error", "status": "OTHER_ERROR"}}
        error_response = Mock()
        error_response.status_code = 401
        error_response.json.return_value = error_data
        error_response.text = json.dumps(error_data)

        # Configure mock to return error
        http_error = requests.exceptions.HTTPError()
        http_error.response = error_response
        error_response.raise_for_status.side_effect = http_error

        mock_get.return_value = error_response

        client = create_test_client()

        # Should raise NestAPIError immediately without retry
        with pytest.raises(NestAPIError) as exc_info:
            client.list_devices()

        assert "401" in str(exc_info.value)

        # Verify GET was called only once (no retry)
        assert mock_get.call_count == 1


class TestListDevices:
    """Test listing devices."""

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    def test_list_devices_success(self, mock_get):
        """Test successfully listing devices."""
        fixture_data = load_fixture("list_devices.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = create_test_client()
        devices = client.list_devices()

        assert len(devices) == 1
        assert devices[0]["name"] == "enterprises/project-id/devices/device-id-1"
        assert devices[0]["traits"]["sdm.devices.traits.ThermostatMode"]["mode"] == "HEAT"

        # Verify the API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "enterprises/test-project/devices" in call_args[0][0]

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    def test_list_devices_empty(self, mock_get):
        """Test listing devices when none exist."""
        fixture_data = load_fixture("empty_device_list.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = create_test_client()
        devices = client.list_devices()

        assert devices == []

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    def test_list_devices_auth_error(self, mock_get):
        """Test authentication error when listing devices."""
        fixture_data = load_fixture("error_unauthenticated.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.status_code = 401
        mock_response.text = "Unauthenticated"

        # Create HTTPError with response attached
        http_error = requests.exceptions.HTTPError()
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        client = create_test_client()

        with pytest.raises(NestAPIError) as exc_info:
            client.list_devices()

        assert "401" in str(exc_info.value)

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    def test_list_devices_permission_error(self, mock_get):
        """Test permission denied error when listing devices."""
        fixture_data = load_fixture("error_permission_denied.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.status_code = 403
        mock_response.text = "Permission denied"

        # Create HTTPError with response attached
        http_error = requests.exceptions.HTTPError()
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        client = create_test_client()

        with pytest.raises(NestAPIError) as exc_info:
            client.list_devices()

        assert "403" in str(exc_info.value)


class TestGetDevice:
    """Test getting a specific device."""

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    def test_get_device_success(self, mock_get):
        """Test successfully getting a device."""
        fixture_data = load_fixture("device_heat_mode.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = create_test_client()
        status = client.get_device("device-id-1")

        assert status.device_id == "enterprises/project-id/devices/device-id-1"
        assert status.mode == "HEAT"
        assert status.heat_setpoint == 20.0

        # Verify the API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "device-id-1" in call_args[0][0]

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    def test_get_device_not_found(self, mock_get):
        """Test getting a non-existent device."""
        fixture_data = load_fixture("error_not_found.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.status_code = 404
        mock_response.text = "Device not found"

        # Create HTTPError with response attached
        http_error = requests.exceptions.HTTPError()
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        client = create_test_client()

        with pytest.raises(NestAPIError) as exc_info:
            client.get_device("nonexistent-device")

        assert "404" in str(exc_info.value)

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    def test_get_device_different_modes(self, mock_get):
        """Test getting devices in different modes."""
        fixtures = [
            ("device_heat_mode.json", "HEAT"),
            ("device_cool_mode.json", "COOL"),
            ("device_heatcool_mode.json", "HEATCOOL"),
            ("device_offline.json", "OFF"),
        ]

        client = create_test_client()

        for fixture_name, expected_mode in fixtures:
            fixture_data = load_fixture(fixture_name)
            mock_response = Mock()
            mock_response.json.return_value = fixture_data
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            status = client.get_device("test-device")
            assert status.mode == expected_mode


class TestSetMode:
    """Test setting thermostat mode."""

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_mode_heat(self, mock_post):
        """Test setting mode to HEAT."""
        fixture_data = load_fixture("command_success_empty.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()
        client.set_mode("device-id-1", ThermostatMode.HEAT)

        # Verify the API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "device-id-1:executeCommand" in call_args[0][0]

        request_json = call_args[1]["json"]
        assert request_json["command"] == "sdm.devices.commands.ThermostatMode.SetMode"
        assert request_json["params"]["mode"] == "HEAT"

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_mode_all_modes(self, mock_post):
        """Test setting all available modes."""
        fixture_data = load_fixture("command_success_empty.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()

        for mode in ThermostatMode:
            client.set_mode("device-id", mode)

            request_json = mock_post.call_args[1]["json"]
            assert request_json["params"]["mode"] == mode.value

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_eco_mode(self, mock_post):
        """Test setting ECO mode."""
        fixture_data = load_fixture("command_success_empty.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()
        client.set_eco_mode("device-id-1", EcoMode.MANUAL_ECO)

        # Verify the API call
        request_json = mock_post.call_args[1]["json"]
        assert request_json["command"] == "sdm.devices.commands.ThermostatEco.SetMode"
        assert request_json["params"]["mode"] == "MANUAL_ECO"


class TestSetTemperature:
    """Test setting temperature setpoints."""

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_heat_temperature(self, mock_post):
        """Test setting heat setpoint."""
        fixture_data = load_fixture("command_success_empty.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()
        client.set_heat("device-id-1", 20.5)

        # Verify the API call
        request_json = mock_post.call_args[1]["json"]
        assert request_json["command"] == "sdm.devices.commands.ThermostatTemperatureSetpoint.SetHeat"
        assert request_json["params"]["heatCelsius"] == 20.5

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_cool_temperature(self, mock_post):
        """Test setting cool setpoint."""
        fixture_data = load_fixture("command_success_empty.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()
        client.set_cool("device-id-1", 24.0)

        # Verify the API call
        request_json = mock_post.call_args[1]["json"]
        assert request_json["command"] == "sdm.devices.commands.ThermostatTemperatureSetpoint.SetCool"
        assert request_json["params"]["coolCelsius"] == 24.0

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_range_temperature(self, mock_post):
        """Test setting temperature range."""
        fixture_data = load_fixture("command_success_empty.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()
        client.set_range("device-id-1", 18.0, 24.0)

        # Verify the API call
        request_json = mock_post.call_args[1]["json"]
        assert request_json["command"] == "sdm.devices.commands.ThermostatTemperatureSetpoint.SetRange"
        assert request_json["params"]["heatCelsius"] == 18.0
        assert request_json["params"]["coolCelsius"] == 24.0

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_temperature_failed_precondition(self, mock_post):
        """Test temperature setting with wrong mode."""
        fixture_data = load_fixture("error_failed_precondition.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.status_code = 400
        mock_response.text = "Cannot set heat setpoint in COOL mode"

        # Create HTTPError with response attached
        http_error = requests.exceptions.HTTPError()
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_post.return_value = mock_response

        client = create_test_client()

        with pytest.raises(NestAPIError) as exc_info:
            client.set_heat("device-id-1", 20.0)

        assert "400" in str(exc_info.value)

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_range_invalid_argument(self, mock_post):
        """Test setting invalid temperature range."""
        fixture_data = load_fixture("error_invalid_argument.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.status_code = 400
        mock_response.text = "Invalid temperature range"

        # Create HTTPError with response attached
        http_error = requests.exceptions.HTTPError()
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_post.return_value = mock_response

        client = create_test_client()

        with pytest.raises(NestAPIError) as exc_info:
            client.set_range("device-id-1", 25.0, 18.0)  # Heat > Cool

        assert "400" in str(exc_info.value)


class TestSetFan:
    """Test setting fan mode."""

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_fan_on(self, mock_post):
        """Test turning fan on."""
        fixture_data = load_fixture("command_success_empty.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()
        client.set_fan("device-id-1", FanMode.ON, duration_seconds=900)

        # Verify the API call
        request_json = mock_post.call_args[1]["json"]
        assert request_json["command"] == "sdm.devices.commands.Fan.SetTimer"
        assert request_json["params"]["timerMode"] == "ON"
        assert request_json["params"]["duration"] == "900s"

    @patch('nest_octopus.nest_thermostat.requests.Session.post')
    def test_set_fan_off(self, mock_post):
        """Test turning fan off."""
        fixture_data = load_fixture("command_success_empty.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = create_test_client()
        client.set_fan("device-id-1", FanMode.OFF)

        # Verify the API call
        request_json = mock_post.call_args[1]["json"]
        assert request_json["command"] == "sdm.devices.commands.Fan.SetTimer"
        assert request_json["params"]["timerMode"] == "OFF"
        assert "duration" not in request_json["params"]


class TestValidFixtures:
    """Test that all valid fixtures can be parsed correctly."""

    def test_all_valid_fixtures_parse(self):
        """Verify all valid fixtures can be loaded and parsed."""
        valid_fixtures = list(VALID_DIR.glob("*.json"))
        assert len(valid_fixtures) > 0, "No valid fixtures found"

        for fixture_path in valid_fixtures:
            with open(fixture_path) as f:
                data = json.load(f)

            # Verify it's valid JSON
            assert isinstance(data, dict)

            # For device fixtures, verify they can be parsed into ThermostatStatus
            if "device_" in fixture_path.name and fixture_path.name != "empty_device_list.json":
                status = ThermostatStatus(data)
                assert status.device_id is not None

    def test_empty_custom_name(self):
        """Test device with empty customName field."""
        data = load_fixture("device_empty_customname.json")
        status = ThermostatStatus(data)

        assert status.device_id == "enterprises/project-id/devices/device-id-empty-name"
        assert status.temperature == 16.769989
        assert status.humidity == 52
        assert status.mode == "HEAT"
        assert status.heat_setpoint == 16.906097

    def test_limited_available_modes(self):
        """Test device with limited available modes (only HEAT and OFF)."""
        data = load_fixture("device_limited_modes.json")
        status = ThermostatStatus(data)

        assert status.device_id == "enterprises/project-id/devices/device-id-limited-modes"
        assert status.mode == "HEAT"
        assert status.hvac_status == "HEATING"
        assert status.heat_setpoint == 19.5

    def test_precise_temperature_values(self):
        """Test device with precise decimal temperature values."""
        data = load_fixture("device_precise_temperatures.json")
        status = ThermostatStatus(data)

        assert status.device_id == "enterprises/project-id/devices/device-id-precise-temps"
        assert status.temperature == 19.876543
        assert status.heat_setpoint == 18.333333
        assert status.cool_setpoint == 21.666667
        assert status.mode == "HEATCOOL"

    def test_real_world_example(self):
        """Test parsing of real API response example."""
        data = load_fixture("device_real_example.json")
        status = ThermostatStatus(data)

        assert status.device_id == "enterprises/2a4g7e4d-8743-1234-5677-73c009871234/devices/abcdefghilmn"
        assert status.type == "sdm.devices.types.THERMOSTAT"
        assert status.temperature == 16.769989
        assert status.humidity == 52
        assert status.mode == "HEAT"
        assert status.eco_mode == "OFF"
        assert status.hvac_status == "OFF"
        assert status.heat_setpoint == 16.906097
        assert status.cool_setpoint is None


class TestInvalidFixtures:
    """Test that invalid fixtures are handled correctly."""

    def test_all_invalid_fixtures_exist(self):
        """Verify all expected invalid fixtures exist."""
        expected_fixtures = [
            "missing_traits.json",
            "error_invalid_argument.json",
            "error_failed_precondition.json",
            "error_unauthenticated.json",
            "error_permission_denied.json",
            "error_not_found.json",
            "missing_device_name.json",
            "invalid_mode.json",
        ]

        for fixture_name in expected_fixtures:
            fixture_path = INVALID_DIR / fixture_name
            assert fixture_path.exists(), f"Missing invalid fixture: {fixture_name}"

    def test_missing_traits_fixture(self):
        """Test device with missing traits."""
        data = load_fixture("missing_traits.json")

        # This should either raise an error or return a status with None values
        # depending on how strictly we validate
        try:
            status = ThermostatStatus(data)
            # If it succeeds, many fields should be None
            assert status.mode is None or status.temperature is None
        except (KeyError, AttributeError):
            # Expected if strict validation
            pass


class TestNoNetworkAccess:
    """Verify that tests never access the network."""

    def test_network_is_blocked(self):
        """Verify that socket access is blocked."""
        with pytest.raises(RuntimeError, match="Network access detected"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    @patch('nest_octopus.nest_thermostat.requests.Session.get')
    def test_requests_are_mocked(self, mock_get):
        """Verify that all requests are mocked."""
        fixture_data = load_fixture("device_heat_mode.json")
        mock_response = Mock()
        mock_response.json.return_value = fixture_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = create_test_client()
        status = client.get_device("device-id")

        # Verify the mock was called
        assert mock_get.called

        # Verify we got data from the fixture, not from network
        assert status.device_id == "enterprises/project-id/devices/device-id-1"
        assert status.mode == "HEAT"
