# SPDX-License-Identifier: MPL-2.0
"""
Google Nest Thermostat API Client Module

This module handles communication with Google Nest thermostats via the
Smart Device Management (SDM) API. It provides methods to query status and
control thermostat settings.

API Documentation: https://developers.google.com/nest/device-access/api/thermostat
"""

import requests
from typing import Dict, Any, Optional
from enum import Enum
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NestAPIError(Exception):
    """Custom exception for Nest API client errors."""
    pass


class ThermostatMode(Enum):
    """Thermostat operating modes."""
    HEAT = "HEAT"
    COOL = "COOL"
    HEATCOOL = "HEATCOOL"
    OFF = "OFF"


class EcoMode(Enum):
    """Eco mode settings."""
    MANUAL_ECO = "MANUAL_ECO"
    OFF = "OFF"


class FanMode(Enum):
    """Fan timer modes."""
    ON = "ON"
    OFF = "OFF"


class ThermostatStatus:
    """
    Represents the current status of a thermostat.

    Attributes:
        device_id (str): Unique device identifier
        temperature (float): Current ambient temperature in Celsius
        humidity (float): Current humidity percentage
        mode (str): Current thermostat mode (HEAT, COOL, HEATCOOL, OFF)
        eco_mode (str): Current eco mode status
        hvac_status (str): Current HVAC system status
        heat_setpoint (float|None): Heat setpoint in Celsius (if applicable)
        cool_setpoint (float|None): Cool setpoint in Celsius (if applicable)
    """

    def __init__(self, device_data: Dict[str, Any]):
        """Initialize from device API response."""
        self.device_id = device_data.get('name', '')
        self.type = device_data.get('type', '')

        traits = device_data.get('traits', {})

        # Temperature trait
        temp_trait = traits.get('sdm.devices.traits.Temperature', {})
        self.temperature = temp_trait.get('ambientTemperatureCelsius')

        # Humidity trait
        humidity_trait = traits.get('sdm.devices.traits.Humidity', {})
        self.humidity = humidity_trait.get('ambientHumidityPercent')

        # Thermostat mode
        mode_trait = traits.get('sdm.devices.traits.ThermostatMode', {})
        self.mode = mode_trait.get('mode')

        # Eco mode
        eco_trait = traits.get('sdm.devices.traits.ThermostatEco', {})
        self.eco_mode = eco_trait.get('mode')

        # HVAC status
        hvac_trait = traits.get('sdm.devices.traits.ThermostatHvac', {})
        self.hvac_status = hvac_trait.get('status')

        # Temperature setpoints
        setpoint_trait = traits.get('sdm.devices.traits.ThermostatTemperatureSetpoint', {})
        self.heat_setpoint = setpoint_trait.get('heatCelsius')
        self.cool_setpoint = setpoint_trait.get('coolCelsius')

    def __repr__(self):
        return (f"ThermostatStatus(temp={self.temperature}°C, "
                f"humidity={self.humidity}%, mode={self.mode}, "
                f"hvac={self.hvac_status})")


class NestThermostatClient:
    """
    Client for interacting with Google Nest thermostats via SDM API.

    Automatically manages OAuth2 token lifecycle using an authorization code.
    On first use, exchanges the authorization code for a refresh token.
    Then automatically refreshes access tokens as needed.

    Attributes:
        base_url (str): The base URL of the SDM API
        project_id (str): Google Cloud project ID
        client_id (str): OAuth 2.0 client ID
        client_secret (str): OAuth 2.0 client secret
        authorization_code (str): OAuth 2.0 authorization code
        redirect_uri (str): OAuth 2.0 redirect URI
        refresh_token (str): OAuth 2.0 refresh token (obtained from auth code)
        refresh_token_expiry (datetime): When the refresh token expires
        timeout (int): Request timeout in seconds
        session (requests.Session): HTTP session for connection pooling
        access_token (str): Current OAuth 2.0 access token (auto-refreshed)
        token_expiry (datetime): When the current access token expires
    """

    BASE_URL = "https://smartdevicemanagement.googleapis.com/v1"
    TOKEN_URL = "https://www.googleapis.com/oauth2/v4/token"

    def __init__(self, project_id: str, refresh_token: str,
                 client_id: str, client_secret: str,
                 timeout: int = 30):
        """
        Initialize the Nest Thermostat API client.

        Args:
            project_id: Google Cloud project ID
            refresh_token: OAuth 2.0 refresh token
            client_id: OAuth 2.0 client ID
            client_secret: OAuth 2.0 client secret
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = self.BASE_URL
        self.project_id = project_id
        self.refresh_token = refresh_token
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

        # Token management
        self.access_token = None
        self.token_expiry = None

        # Get initial access token
        self._ensure_valid_token()

    def _refresh_access_token(self):
        """
        Refresh the OAuth2 access token using the refresh token.

        Raises:
            NestAPIError: If token refresh fails
        """
        try:
            logger.info("Refreshing OAuth2 access token")

            params = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'refresh_token': self.refresh_token,
                'grant_type': 'refresh_token'
            }

            response = requests.post(self.TOKEN_URL, params=params, timeout=self.timeout)
            response.raise_for_status()

            token_data = response.json()

            # Extract access token (required field)
            if 'access_token' not in token_data:
                raise KeyError("access_token not found in response")
            self.access_token = token_data['access_token']

            # Set expiry time (default to 3600 seconds if not provided)
            expires_in = token_data.get('expires_in', 3600)
            # Refresh 5 minutes before actual expiry to avoid race conditions
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)

            # Log additional token info if present
            if 'token_type' in token_data:
                logger.debug(f"Token type: {token_data['token_type']}")
            if 'scope' in token_data:
                logger.debug(f"Token scope: {token_data['scope']}")
            if 'refresh_token_expires_in' in token_data:
                logger.info(f"Refresh token expires in {token_data['refresh_token_expires_in']} seconds")

            # Update session authorization header
            self.session.headers.update({
                'Authorization': f'Bearer {self.access_token}'
            })

            logger.info(f"Access token refreshed, expires at {self.token_expiry}")

        except requests.exceptions.HTTPError as e:
            error_msg = f"OAuth2 token refresh failed: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except (KeyError, ValueError) as e:
            error_msg = f"Invalid token response: {str(e)}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Token refresh request failed: {str(e)}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

    def _ensure_valid_token(self):
        """
        Ensure access token is valid, refreshing if necessary.
        """
        if self.token_expiry is None or datetime.now() >= self.token_expiry:
            logger.info("Access token expired or about to expire, refreshing")
            self._refresh_access_token()

    def _is_token_expired_error(self, response) -> bool:
        """
        Check if an HTTP error response indicates an expired/invalid token.

        Args:
            response: The HTTP response object

        Returns:
            True if the error is due to expired/invalid authentication
        """
        if response.status_code != 401:
            return False

        try:
            error_data = response.json()
            error_info = error_data.get('error', {})
            return error_info.get('status') == 'UNAUTHENTICATED'
        except (ValueError, KeyError):
            return False

    def list_devices(self) -> list[Dict[str, Any]]:
        """
        List all devices in the project.

        Returns:
            List of device data dictionaries

        Raises:
            NestAPIError: If the request fails
        """
        self._ensure_valid_token()
        url = f"{self.base_url}/enterprises/{self.project_id}/devices"

        try:
            logger.info(f"Listing devices for project {self.project_id}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            devices = data.get('devices', [])
            logger.info(f"Found {len(devices)} device(s)")
            return devices

        except requests.exceptions.HTTPError as e:
            # If token expired during request, refresh and retry once
            if self._is_token_expired_error(e.response):
                logger.warning("Token expired during request, refreshing and retrying")
                self._refresh_access_token()
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()
                    devices = data.get('devices', [])
                    logger.info(f"Found {len(devices)} device(s) after retry")
                    return devices
                except requests.exceptions.HTTPError as retry_error:
                    error_msg = f"HTTP error after retry: {retry_error.response.status_code} - {retry_error.response.text}"
                    logger.error(error_msg)
                    raise NestAPIError(error_msg)

            error_msg = f"HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except requests.exceptions.Timeout:
            error_msg = f"Request timed out after {self.timeout} seconds"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except ValueError as e:
            error_msg = f"Invalid JSON response: {str(e)}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

    def get_device(self, device_id: str) -> ThermostatStatus:
        """
        Get detailed information about a specific device.

        Args:
            device_id: Full device ID (enterprises/{project}/devices/{device})

        Returns:
            ThermostatStatus object with current device state

        Raises:
            NestAPIError: If the request fails
        """
        self._ensure_valid_token()
        url = f"{self.base_url}/{device_id}"

        try:
            logger.info(f"Getting device info for {device_id}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            status = ThermostatStatus(data)
            logger.info(f"Retrieved status: {status}")
            return status

        except requests.exceptions.HTTPError as e:
            # If token expired during request, refresh and retry once
            if self._is_token_expired_error(e.response):
                logger.warning("Token expired during request, refreshing and retrying")
                self._refresh_access_token()
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    data = response.json()
                    status = ThermostatStatus(data)
                    logger.info(f"Retrieved status after retry: {status}")
                    return status
                except requests.exceptions.HTTPError as retry_error:
                    error_msg = f"HTTP error after retry: {retry_error.response.status_code} - {retry_error.response.text}"
                    logger.error(error_msg)
                    raise NestAPIError(error_msg)

            error_msg = f"HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except requests.exceptions.Timeout:
            error_msg = f"Request timed out after {self.timeout} seconds"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except (ValueError, KeyError) as e:
            error_msg = f"Invalid response data: {str(e)}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

    def set_mode(self, device_id: str, mode: ThermostatMode) -> Dict[str, Any]:
        """
        Set the thermostat operating mode.

        Args:
            device_id: Full device ID
            mode: Target thermostat mode (HEAT, COOL, HEATCOOL, or OFF)

        Returns:
            API response dictionary

        Raises:
            NestAPIError: If the request fails
        """
        command = "sdm.devices.commands.ThermostatMode.SetMode"
        params = {"mode": mode.value}

        return self._execute_command(device_id, command, params)

    def set_eco_mode(self, device_id: str, eco_mode: EcoMode) -> Dict[str, Any]:
        """
        Set the thermostat eco mode.

        Args:
            device_id: Full device ID
            eco_mode: Target eco mode (MANUAL_ECO or OFF)

        Returns:
            API response dictionary

        Raises:
            NestAPIError: If the request fails
        """
        command = "sdm.devices.commands.ThermostatEco.SetMode"
        params = {"mode": eco_mode.value}

        return self._execute_command(device_id, command, params)

    def set_heat(self, device_id: str, temperature_celsius: float) -> Dict[str, Any]:
        """
        Set the heat setpoint (thermostat must be in HEAT mode).

        Args:
            device_id: Full device ID
            temperature_celsius: Target temperature in Celsius

        Returns:
            API response dictionary

        Raises:
            NestAPIError: If the request fails or thermostat not in HEAT mode
        """
        command = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetHeat"
        params = {"heatCelsius": temperature_celsius}

        return self._execute_command(device_id, command, params)

    def set_cool(self, device_id: str, temperature_celsius: float) -> Dict[str, Any]:
        """
        Set the cool setpoint (thermostat must be in COOL mode).

        Args:
            device_id: Full device ID
            temperature_celsius: Target temperature in Celsius

        Returns:
            API response dictionary

        Raises:
            NestAPIError: If the request fails or thermostat not in COOL mode
        """
        command = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetCool"
        params = {"coolCelsius": temperature_celsius}

        return self._execute_command(device_id, command, params)

    def set_range(self, device_id: str, heat_celsius: float,
                  cool_celsius: float) -> Dict[str, Any]:
        """
        Set the heat and cool setpoints (thermostat must be in HEATCOOL mode).

        Args:
            device_id: Full device ID
            heat_celsius: Heat setpoint in Celsius
            cool_celsius: Cool setpoint in Celsius (must be > heat_celsius)

        Returns:
            API response dictionary

        Raises:
            NestAPIError: If the request fails or invalid temperature range
        """
        command = "sdm.devices.commands.ThermostatTemperatureSetpoint.SetRange"
        params = {
            "heatCelsius": heat_celsius,
            "coolCelsius": cool_celsius
        }

        return self._execute_command(device_id, command, params)

    def set_fan(self, device_id: str, mode: FanMode,
                duration_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Set the fan timer.

        Args:
            device_id: Full device ID
            mode: FanMode.ON or FanMode.OFF
            duration_seconds: How long to run fan (default 900s/15min if ON)

        Returns:
            API response dictionary

        Raises:
            NestAPIError: If the request fails or thermostat has no fan
        """
        command = "sdm.devices.commands.Fan.SetTimer"
        params = {"timerMode": mode.value}

        if mode == FanMode.ON and duration_seconds is not None:
            params["duration"] = f"{duration_seconds}s"

        return self._execute_command(device_id, command, params)

    def _execute_command(self, device_id: str, command: str,
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a command on the device.

        Args:
            device_id: Full device ID
            command: Command string (e.g., 'sdm.devices.commands.ThermostatMode.SetMode')
            params: Command parameters

        Returns:
            API response dictionary

        Raises:
            NestAPIError: If the command fails
        """
        self._ensure_valid_token()
        url = f"{self.base_url}/{device_id}:executeCommand"
        payload = {
            "command": command,
            "params": params
        }

        try:
            logger.info(f"Executing command {command} on {device_id} with params {params}")
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            # Commands may return empty response on success
            if response.text:
                result = response.json()
            else:
                result = {}

            logger.info(f"Command executed successfully")
            return result

        except requests.exceptions.HTTPError as e:
            # If token expired during request, refresh and retry once
            if self._is_token_expired_error(e.response):
                logger.warning("Token expired during command execution, refreshing and retrying")
                self._refresh_access_token()
                try:
                    response = self.session.post(url, json=payload, timeout=self.timeout)
                    response.raise_for_status()
                    if response.text:
                        result = response.json()
                    else:
                        result = {}
                    logger.info(f"Command executed successfully after retry")
                    return result
                except requests.exceptions.HTTPError as retry_error:
                    error_msg = f"HTTP error after retry: {retry_error.response.status_code} - {retry_error.response.text}"
                    logger.error(error_msg)
                    raise NestAPIError(error_msg)

            error_msg = f"HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except requests.exceptions.Timeout:
            error_msg = f"Request timed out after {self.timeout} seconds"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

        except ValueError as e:
            error_msg = f"Invalid JSON response: {str(e)}"
            logger.error(error_msg)
            raise NestAPIError(error_msg)

    def close(self):
        """Close the HTTP session."""
        self.session.close()
        logger.info("Nest API client session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
