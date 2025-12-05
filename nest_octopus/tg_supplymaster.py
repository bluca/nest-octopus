# SPDX-License-Identifier: MPL-2.0
"""
Timeguard SupplyMaster WiFi Power Switch Client

API client for controlling Timeguard FSTWIFI Wi-Fi Controlled Fused Spur devices.
Supports device management, scheduling programs, and mode control.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, cast

import requests

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkMode(Enum):
    """Device work modes."""
    AUTO = "0"  # Auto timed (use the active program/schedule)
    ALWAYS_OFF = "1"  # Always Off
    ALWAYS_ON = "2"  # Always On
    HOLIDAY = "3"  # Holiday mode


class DayOfWeek(Enum):
    """Days of the week mapping."""
    SUNDAY = 0
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6


@dataclass
class TimeSlot:
    """Represents a start or end time for a program slot."""
    enable: bool
    time: str  # Format: "HH:MM"

    def to_dict(self) -> Dict[str, str]:
        """Convert to API format."""
        return {
            "enable": "1" if self.enable else "0",
            "time": self.time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "TimeSlot":
        """Create from API response."""
        return cls(
            enable=data["enable"] == "1",
            time=data["time"]
        )


@dataclass
class ProgramSlot:
    """Represents one on/off time slot in a program."""
    start: TimeSlot
    end: TimeSlot
    days: Dict[DayOfWeek, bool]  # Which days this slot is active

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "map": {str(day.value): "1" if enabled else "0" for day, enabled in self.days.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgramSlot":
        """Create from API response."""
        days = {
            DayOfWeek(int(k)): v == "1"
            for k, v in data["map"].items()
        }
        return cls(
            start=TimeSlot.from_dict(data["start"]),
            end=TimeSlot.from_dict(data["end"]),
            days=days
        )


@dataclass
class Program:
    """Represents a complete program (schedule) with up to 6 time slots."""
    id: str
    name: str
    slots: List[ProgramSlot]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        return {
            "id": self.id,
            "name": self.name,
            "program": [slot.to_dict() for slot in self.slots]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from API response."""
        return cls(
            id=data["id"],
            name=data["name"],
            slots=[ProgramSlot.from_dict(slot) for slot in data["program"]]
        )


@dataclass
class Holiday:
    """Holiday mode configuration."""
    enable: bool
    start: datetime
    end: datetime

    def to_dict(self) -> Dict[str, str]:
        """Convert to API format."""
        return {
            "enable": "1" if self.enable else "0",
            "start": self.start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": self.end.strftime("%Y-%m-%d %H:%M:%S")
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Holiday":
        """Create from API response."""
        return cls(
            enable=data["enable"] == "1",
            start=datetime.strptime(data["start"], "%Y-%m-%d %H:%M:%S"),
            end=datetime.strptime(data["end"], "%Y-%m-%d %H:%M:%S")
        )


@dataclass
class DeviceInfo:
    """Device information."""
    device_id: str
    name: str
    online: bool
    relay: bool  # Current relay state
    work_mode: WorkMode
    advance: bool
    loaded: bool  # Main relay loaded
    work_status: bool  # Main relay work status

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceInfo":
        """Create from API response."""
        return cls(
            device_id=data["device_id"],
            name=data["name"],
            online=data["online"] == "1",
            relay=data["relay"] == "1",
            work_mode=WorkMode(str(data["work_mode"])),
            advance=data["advance"] == "1",
            loaded=data["main_relay"]["loaded"] == "1",
            work_status=data["main_relay"]["work_status"] == "1"
        )


class SupplyMasterError(Exception):
    """Base exception for SupplyMaster errors."""
    pass


class AuthenticationError(SupplyMasterError):
    """Authentication failed."""
    pass


class DeviceNotFoundError(SupplyMasterError):
    """Device not found."""
    pass


class SupplyMasterClient:
    """
    Client for Timeguard SupplyMaster API.

    Provides methods to control WiFi power switches, manage programs,
    and configure device settings.
    """

    BASE_URL = "https://www.cloudwarm.net/TimeGuard/api/Android/v_1"
    USER_AGENT = "okhttp/3.3.1"

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        user_id: Optional[str] = None,
        token: Optional[str] = None,
        device_id: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize the SupplyMaster client.

        Args:
            username: Account username (for authentication)
            password: Account password (for authentication)
            user_id: User ID (if already authenticated)
            token: Authentication token (if already authenticated)
            device_id: Device ID to control
            timeout: Request timeout in seconds
        """
        self.user_id = user_id
        self.token = token
        self.device_id = device_id
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.USER_AGENT})

        # Authenticate if credentials provided
        if username and password:
            self.authenticate(username, password)

    def __enter__(self) -> "SupplyMasterClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate and obtain access token.

        Args:
            username: Account username
            password: Account password

        Returns:
            Authentication response containing user info and devices

        Raises:
            AuthenticationError: If authentication fails
        """
        url = f"{self.BASE_URL}/users/login"
        data = {"username": username, "password": password}

        try:
            response = self._session.put(url, data=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise AuthenticationError(f"Authentication failed: {result.get('message', 'Unknown error')}")

            message = result["message"]
            user = message["user"]
            self.user_id = user["id"]
            self.token = user["token"]

            # Auto-select device if only one exists
            devices = message.get("wifi_box", [])
            if len(devices) == 1:
                self.device_id = devices[0]["device_id"]
                logger.debug(f"Auto-selected device: {self.device_id}")

            return cast(Dict[str, Any], result)

        except requests.RequestException as e:
            raise AuthenticationError(f"Authentication request failed: {e}")

    def list_devices(self) -> List[DeviceInfo]:
        """
        List all devices associated with the account.

        Returns:
            List of DeviceInfo objects

        Raises:
            AuthenticationError: If not authenticated
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()

        url = f"{self.BASE_URL}/users/wifi_boxes/user_id/{self.user_id}/is_sub_user/0/token/{self.token}"

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to list devices: {result.get('message', 'Unknown error')}")

            return [DeviceInfo.from_dict(device) for device in result["message"]]

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to list devices: {e}")

    def get_device_info(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed device information.

        Args:
            device_id: Device ID (uses instance device_id if not provided)

        Returns:
            Device information dictionary

        Raises:
            DeviceNotFoundError: If device_id not set
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/data/user_id/{self.user_id}/wifi_box_id/{device_id}/token/{self.token}"

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to get device info: {result.get('message', 'Unknown error')}")

            return cast(Dict[str, Any], result["message"])

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to get device info: {e}")

    def set_device_name(self, name: str, device_id: Optional[str] = None) -> None:
        """
        Set device name.

        Name requirements:
        - Must start with a letter
        - Must be between 5 to 19 characters
        - Allowed: Letters, numbers, and spaces

        Args:
            name: New device name
            device_id: Device ID (uses instance device_id if not provided)

        Raises:
            ValueError: If name doesn't meet requirements
            SupplyMasterError: If request fails
        """
        if not (5 <= len(name) <= 19):
            raise ValueError("Name must be between 5 and 19 characters")
        if not name[0].isalpha():
            raise ValueError("Name must start with a letter")

        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/name"
        data = {
            "token": self.token,
            "user_id": self.user_id,
            "wifi_box_id": device_id,
            "name": name
        }

        try:
            response = self._session.put(url, data=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to set device name: {result.get('message', 'Unknown error')}")

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to set device name: {e}")

    def get_mode(self, device_id: Optional[str] = None) -> tuple[WorkMode, Holiday]:
        """
        Get current work mode and holiday settings.

        Args:
            device_id: Device ID (uses instance device_id if not provided)

        Returns:
            Tuple of (WorkMode, Holiday)

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/mode_and_holiday/user_id/{self.user_id}/wifi_box_id/{device_id}/token/{self.token}"

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to get mode: {result.get('message', 'Unknown error')}")

            message = result["message"]
            return (
                WorkMode(message["work_mode"]),
                Holiday.from_dict(message["holiday"])
            )

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to get mode: {e}")

    def set_mode(self, mode: WorkMode, device_id: Optional[str] = None) -> None:
        """
        Set device work mode.

        Args:
            mode: Work mode to set
            device_id: Device ID (uses instance device_id if not provided)

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/work_mode"
        data = {
            "work_mode": mode.value,
            "token": self.token,
            "user_id": self.user_id,
            "wifi_box_id": device_id
        }

        try:
            response = self._session.put(url, data=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to set mode: {result.get('message', 'Unknown error')}")

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to set mode: {e}")

    def set_boost(self, enable: bool = True, device_id: Optional[str] = None) -> None:
        """
        Enable/disable boost mode (turns on device for 1 hour).

        Args:
            enable: True to enable boost, False to disable
            device_id: Device ID (uses instance device_id if not provided)

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/boost"
        data = {
            "boost": "1" if enable else "0",
            "token": self.token,
            "user_id": self.user_id,
            "wifi_box_id": device_id
        }

        try:
            response = self._session.put(url, data=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to set boost: {result.get('message', 'Unknown error')}")

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to set boost: {e}")

    def set_advance(self, enable: bool = True, device_id: Optional[str] = None) -> None:
        """
        Enable/disable advance mode (turns on device until next scheduled off time).

        Args:
            enable: True to enable advance, False to disable
            device_id: Device ID (uses instance device_id if not provided)

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/advance"
        data = {
            "advance": "1" if enable else "0",
            "token": self.token,
            "user_id": self.user_id,
            "wifi_box_id": device_id
        }

        try:
            response = self._session.put(url, data=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to set advance: {result.get('message', 'Unknown error')}")

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to set advance: {e}")

    def list_programs(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """
        List all programs (schedules) for the device.

        Args:
            device_id: Device ID (uses instance device_id if not provided)

        Returns:
            Dictionary with 'choosex' (active program) and 'namelist'

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/program_list/user_id/{self.user_id}/wifi_box_id/{device_id}/token/{self.token}"

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to list programs: {result.get('message', 'Unknown error')}")

            return cast(Dict[str, Any], result["message"])

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to list programs: {e}")

    def get_program(self, program_id: str, device_id: Optional[str] = None) -> Program:
        """
        Get a specific program by ID.

        Args:
            program_id: Program ID (0-5)
            device_id: Device ID (uses instance device_id if not provided)

        Returns:
            Program object

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/program/user_id/{self.user_id}/wifi_box_id/{device_id}/index/{program_id}/token/{self.token}"

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to get program: {result.get('message', 'Unknown error')}")

            return Program.from_dict(result["message"])

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to get program: {e}")

    def update_program(self, program: Program, device_id: Optional[str] = None) -> None:
        """
        Update a program (schedule).

        Args:
            program: Program object to update
            device_id: Device ID (uses instance device_id if not provided)

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/program"
        data = {
            "token": self.token,
            "program": json.dumps(program.to_dict()),
            "user_id": self.user_id,
            "wifi_box_id": device_id
        }

        try:
            response = self._session.put(url, data=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to update program: {result.get('message', 'Unknown error')}")

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to update program: {e}")

    def enable_program(self, program_id: str, device_id: Optional[str] = None) -> None:
        """
        Enable (activate) a specific program.

        Args:
            program_id: Program ID (0-5)
            device_id: Device ID (uses instance device_id if not provided)

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/program_enable"
        data = {
            "token": self.token,
            "index": program_id,
            "user_id": self.user_id,
            "wifi_box_id": device_id
        }

        try:
            response = self._session.put(url, data=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to enable program: {result.get('message', 'Unknown error')}")

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to enable program: {e}")

    def set_program_name(self, program_id: str, name: str, device_id: Optional[str] = None) -> None:
        """
        Set the name of a program.

        Args:
            program_id: Program ID (0-5)
            name: New program name
            device_id: Device ID (uses instance device_id if not provided)

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/program_name"
        para = json.dumps({"name": name, "id": program_id})
        data = {
            "token": self.token,
            "user_id": self.user_id,
            "para": para,
            "wifi_box_id": device_id
        }

        try:
            response = self._session.put(url, data=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to set program name: {result.get('message', 'Unknown error')}")

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to set program name: {e}")

    def set_holiday(self, holiday: Holiday, device_id: Optional[str] = None) -> None:
        """
        Set holiday mode configuration.

        Note: If enable=True, this will also change device mode to Holiday.

        Args:
            holiday: Holiday configuration
            device_id: Device ID (uses instance device_id if not provided)

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/holiday"
        data = {
            "holiday": json.dumps(holiday.to_dict()),
            "token": self.token,
            "user_id": self.user_id,
            "wifi_box_id": device_id
        }

        try:
            response = self._session.put(url, data=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to set holiday: {result.get('message', 'Unknown error')}")

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to set holiday: {e}")

    def get_holiday(self, device_id: Optional[str] = None) -> Holiday:
        """
        Get holiday mode configuration.

        Args:
            device_id: Device ID (uses instance device_id if not provided)

        Returns:
            Holiday object

        Raises:
            SupplyMasterError: If request fails
        """
        self._ensure_authenticated()
        device_id = device_id or self.device_id
        if not device_id:
            raise DeviceNotFoundError("No device_id specified")

        url = f"{self.BASE_URL}/wifi_boxes/holiday/user_id/{self.user_id}/wifi_box_id/{device_id}/token/{self.token}"

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            if not result.get("status"):
                raise SupplyMasterError(f"Failed to get holiday: {result.get('message', 'Unknown error')}")

            return Holiday.from_dict(result["message"])

        except requests.RequestException as e:
            raise SupplyMasterError(f"Failed to get holiday: {e}")

    def _ensure_authenticated(self) -> None:
        """Ensure user is authenticated."""
        if not self.user_id or not self.token:
            raise AuthenticationError("Not authenticated. Call authenticate() first or provide user_id and token.")
