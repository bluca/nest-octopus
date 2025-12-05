# SPDX-License-Identifier: MPL-2.0
"""Unit tests for TG SupplyMaster client."""

import json
from datetime import datetime
from unittest.mock import Mock, patch

from typing import Any, Generator

import pytest
import requests

from nest_octopus.tg_supplymaster import (
    AuthenticationError,
    DayOfWeek,
    DeviceInfo,
    DeviceNotFoundError,
    Holiday,
    Program,
    ProgramSlot,
    SupplyMasterClient,
    SupplyMasterError,
    TimeSlot,
    WorkMode,
)


class TestTimeSlot:
    """Tests for TimeSlot dataclass."""

    def test_to_dict_enabled(self) -> None:
        slot = TimeSlot(enable=True, time="08:30")
        assert slot.to_dict() == {"enable": "1", "time": "08:30"}

    def test_to_dict_disabled(self) -> None:
        slot = TimeSlot(enable=False, time="00:00")
        assert slot.to_dict() == {"enable": "0", "time": "00:00"}

    def test_from_dict_enabled(self) -> None:
        data = {"enable": "1", "time": "15:45"}
        slot = TimeSlot.from_dict(data)
        assert slot.enable is True
        assert slot.time == "15:45"

    def test_from_dict_disabled(self) -> None:
        data = {"enable": "0", "time": "00:00"}
        slot = TimeSlot.from_dict(data)
        assert slot.enable is False
        assert slot.time == "00:00"


class TestProgramSlot:
    """Tests for ProgramSlot dataclass."""

    def test_to_dict(self) -> None:
        days = {day: True for day in DayOfWeek}
        days[DayOfWeek.MONDAY] = False

        slot = ProgramSlot(
            start=TimeSlot(True, "08:00"),
            end=TimeSlot(True, "17:00"),
            days=days
        )

        result = slot.to_dict()
        assert result["start"] == {"enable": "1", "time": "08:00"}
        assert result["end"] == {"enable": "1", "time": "17:00"}
        assert result["map"]["0"] == "1"  # Sunday
        assert result["map"]["1"] == "0"  # Monday
        assert result["map"]["2"] == "1"  # Tuesday

    def test_from_dict(self) -> None:
        data = {
            "start": {"enable": "1", "time": "09:00"},
            "end": {"enable": "1", "time": "18:00"},
            "map": {
                "0": "1",
                "1": "0",
                "2": "1",
                "3": "1",
                "4": "1",
                "5": "1",
                "6": "1"
            }
        }

        slot = ProgramSlot.from_dict(data)
        assert slot.start.time == "09:00"
        assert slot.end.time == "18:00"
        assert slot.days[DayOfWeek.SUNDAY] is True
        assert slot.days[DayOfWeek.MONDAY] is False
        assert slot.days[DayOfWeek.TUESDAY] is True


class TestProgram:
    """Tests for Program dataclass."""

    def test_to_dict(self) -> None:
        days_all = {day: True for day in DayOfWeek}
        slot1 = ProgramSlot(
            start=TimeSlot(True, "08:00"),
            end=TimeSlot(True, "09:00"),
            days=days_all
        )

        program = Program(id="0", name="Test Program", slots=[slot1])
        result = program.to_dict()

        assert result["id"] == "0"
        assert result["name"] == "Test Program"
        assert len(result["program"]) == 1
        assert result["program"][0]["start"]["time"] == "08:00"

    def test_from_dict(self) -> None:
        data = {
            "id": "1",
            "name": "Morning",
            "program": [
                {
                    "start": {"enable": "1", "time": "06:00"},
                    "end": {"enable": "1", "time": "08:00"},
                    "map": {str(i): "1" for i in range(7)}
                }
            ]
        }

        program = Program.from_dict(data)
        assert program.id == "1"
        assert program.name == "Morning"
        assert len(program.slots) == 1
        assert program.slots[0].start.time == "06:00"


class TestHoliday:
    """Tests for Holiday dataclass."""

    def test_to_dict(self) -> None:
        holiday = Holiday(
            enable=True,
            start=datetime(2025, 12, 24, 0, 0, 0),
            end=datetime(2026, 1, 2, 23, 59, 59)
        )
        result = holiday.to_dict()

        assert result["enable"] == "1"
        assert result["start"] == "2025-12-24 00:00:00"
        assert result["end"] == "2026-01-02 23:59:59"

    def test_from_dict(self) -> None:
        data = {
            "enable": "1",
            "start": "2025-07-01 00:00:00",
            "end": "2025-08-31 23:59:59"
        }

        holiday = Holiday.from_dict(data)
        assert holiday.enable is True
        assert holiday.start == datetime(2025, 7, 1, 0, 0, 0)
        assert holiday.end == datetime(2025, 8, 31, 23, 59, 59)


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_from_dict(self) -> None:
        data = {
            "device_id": "0123456789",
            "name": "My Device",
            "online": "1",
            "relay": "1",
            "work_mode": 0,  # API returns integers
            "advance": "0",
            "main_relay": {
                "loaded": "1",
                "work_status": "0"
            }
        }

        info = DeviceInfo.from_dict(data)
        assert info.device_id == "0123456789"
        assert info.name == "My Device"
        assert info.online is True
        assert info.relay is True
        assert info.work_mode == WorkMode.AUTO
        assert info.advance is False
        assert info.loaded is True
        assert info.work_status is False


class TestSupplyMasterClient:
    """Tests for SupplyMasterClient."""

    @pytest.fixture
    def mock_session(self) -> Generator[Any, None, None]:
        """Create a mock session."""
        with patch('nest_octopus.tg_supplymaster.requests.Session') as mock:
            session = Mock()
            mock.return_value = session
            yield session

    def test_init_no_auth(self, mock_session: Any) -> None:
        """Test initialization without authentication."""
        client = SupplyMasterClient()
        assert client.user_id is None
        assert client.token is None
        assert client.device_id is None
        assert client.timeout == 30

    def test_init_with_credentials(self, mock_session: Any) -> None:
        """Test initialization with username/password."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": {
                "user": {
                    "id": "12345",
                    "token": "test-token"
                },
                "wifi_box": [
                    {"device_id": "0123456789", "name": "Device"}
                ]
            },
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient(username="test", password="pass")
        assert client.user_id == "12345"
        assert client.token == "test-token"
        assert client.device_id == "0123456789"

    def test_context_manager(self, mock_session: Any) -> None:
        """Test context manager usage."""
        with SupplyMasterClient(user_id="123", token="abc") as client:
            assert client.user_id == "123"
        mock_session.close.assert_called_once()

    def test_authenticate_success(self, mock_session: Any) -> None:
        """Test successful authentication."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": {
                "user": {
                    "id": "12345",
                    "username": "testuser",
                    "token": "secret-token"
                },
                "wifi_box": [
                    {"device_id": "0123456789", "name": "My Device"}
                ]
            },
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient()
        result = client.authenticate("testuser", "password123")

        assert client.user_id == "12345"
        assert client.token == "secret-token"
        assert client.device_id == "0123456789"
        assert result["status"] is True

    def test_authenticate_failure(self, mock_session: Any) -> None:
        """Test failed authentication."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": False,
            "message": "Invalid credentials",
            "error_code": 1
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient()
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            client.authenticate("baduser", "badpass")

    def test_authenticate_network_error(self, mock_session: Any) -> None:
        """Test authentication with network error."""
        mock_session.put.side_effect = requests.RequestException("Connection failed")

        client = SupplyMasterClient()
        with pytest.raises(AuthenticationError, match="Authentication request failed"):
            client.authenticate("user", "pass")

    def test_list_devices(self, mock_session: Any) -> None:
        """Test listing devices."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": [
                {
                    "device_id": "device1",
                    "name": "Device 1",
                    "online": "1",
                    "relay": "1",
                    "work_mode": 0,
                    "advance": "0",
                    "main_relay": {"loaded": "1", "work_status": "1"}
                },
                {
                    "device_id": "device2",
                    "name": "Device 2",
                    "online": "1",
                    "relay": "0",
                    "work_mode": 1,
                    "advance": "1",
                    "main_relay": {"loaded": "0", "work_status": "0"}
                }
            ],
            "error_code": 0
        }
        mock_session.get.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc")
        devices = client.list_devices()

        assert len(devices) == 2
        assert devices[0].device_id == "device1"
        assert devices[0].name == "Device 1"
        assert devices[0].online is True
        assert devices[1].device_id == "device2"
        assert devices[1].name == "Device 2"
        assert devices[1].online is True

    def test_list_devices_not_authenticated(self) -> None:
        """Test listing devices without authentication."""
        client = SupplyMasterClient()
        with pytest.raises(AuthenticationError, match="Not authenticated"):
            client.list_devices()

    def test_get_device_info(self, mock_session: Any) -> None:
        """Test getting device info."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": {
                "relay": "1",
                "time_zone": "0",
                "pro_index": "0",
                "work_mode": "0",
                "advance": "0",
                "main_relay": {"loaded": "1", "work_status": "1"}
            },
            "error_code": 0
        }
        mock_session.get.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        info = client.get_device_info()

        assert info["relay"] == "1"
        assert info["work_mode"] == "0"

    def test_get_device_info_no_device_id(self) -> None:
        """Test getting device info without device_id."""
        client = SupplyMasterClient(user_id="123", token="abc")
        with pytest.raises(DeviceNotFoundError, match="No device_id specified"):
            client.get_device_info()

    def test_set_device_name_valid(self, mock_session: Any) -> None:
        """Test setting valid device name."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        client.set_device_name("MyDevice")  # Valid: 8 chars

        # Should not raise

    def test_set_device_name_too_short(self) -> None:
        """Test setting device name that's too short."""
        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        with pytest.raises(ValueError, match="between 5 and 19 characters"):
            client.set_device_name("Dev")  # Only 3 chars

    def test_set_device_name_too_long(self) -> None:
        """Test setting device name that's too long."""
        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        with pytest.raises(ValueError, match="between 5 and 19 characters"):
            client.set_device_name("A" * 20)  # 20 chars

    def test_set_device_name_invalid_start(self) -> None:
        """Test setting device name that doesn't start with letter."""
        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        with pytest.raises(ValueError, match="must start with a letter"):
            client.set_device_name("1Device")  # Starts with number

    def test_get_mode(self, mock_session: Any) -> None:
        """Test getting work mode."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": {
                "work_mode": "2",
                "holiday": {
                    "enable": "0",
                    "start": "1970-01-01 00:00:00",
                    "end": "1970-01-01 00:00:00"
                }
            },
            "error_code": 0
        }
        mock_session.get.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        mode, holiday = client.get_mode()

        assert mode == WorkMode.ALWAYS_ON
        assert holiday.enable is False

    def test_set_mode(self, mock_session: Any) -> None:
        """Test setting work mode."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        client.set_mode(WorkMode.ALWAYS_OFF)

        # Verify the correct data was sent
        call_args = mock_session.put.call_args
        assert call_args[1]["data"]["work_mode"] == "1"

    def test_set_boost_enable(self, mock_session: Any) -> None:
        """Test enabling boost mode."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        client.set_boost(True)

        call_args = mock_session.put.call_args
        assert call_args[1]["data"]["boost"] == "1"

    def test_set_boost_disable(self, mock_session: Any) -> None:
        """Test disabling boost mode."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        client.set_boost(False)

        call_args = mock_session.put.call_args
        assert call_args[1]["data"]["boost"] == "0"

    def test_set_advance(self, mock_session: Any) -> None:
        """Test setting advance mode."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        client.set_advance(True)

        call_args = mock_session.put.call_args
        assert call_args[1]["data"]["advance"] == "1"

    def test_list_programs(self, mock_session: Any) -> None:
        """Test listing programs."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": {
                "choosex": "0",
                "namelist": [
                    {"id": "0", "name": "Morning"},
                    {"id": "1", "name": "Evening"},
                    {"id": "2", "name": ""},
                ]
            },
            "error_code": 0
        }
        mock_session.get.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        result = client.list_programs()

        assert result["choosex"] == "0"
        assert len(result["namelist"]) == 3
        assert result["namelist"][0]["name"] == "Morning"

    def test_get_program(self, mock_session: Any) -> None:
        """Test getting a specific program."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": {
                "id": "0",
                "name": "Default",
                "program": [
                    {
                        "start": {"enable": "1", "time": "08:00"},
                        "end": {"enable": "1", "time": "17:00"},
                        "map": {str(i): "1" for i in range(7)}
                    }
                ]
            },
            "error_code": 0
        }
        mock_session.get.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        program = client.get_program("0")

        assert program.id == "0"
        assert program.name == "Default"
        assert len(program.slots) == 1
        assert program.slots[0].start.time == "08:00"

    def test_update_program(self, mock_session: Any) -> None:
        """Test updating a program."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        days_all = {day: True for day in DayOfWeek}
        slot = ProgramSlot(
            start=TimeSlot(True, "09:00"),
            end=TimeSlot(True, "18:00"),
            days=days_all
        )
        program = Program(id="1", name="Work Hours", slots=[slot])

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        client.update_program(program)

        # Verify JSON encoding
        call_args = mock_session.put.call_args
        program_data = json.loads(call_args[1]["data"]["program"])
        assert program_data["id"] == "1"
        assert program_data["name"] == "Work Hours"

    def test_enable_program(self, mock_session: Any) -> None:
        """Test enabling a program."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        client.enable_program("2")

        call_args = mock_session.put.call_args
        assert call_args[1]["data"]["index"] == "2"

    def test_set_program_name(self, mock_session: Any) -> None:
        """Test setting program name."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        client.set_program_name("1", "New Name")

        call_args = mock_session.put.call_args
        para = json.loads(call_args[1]["data"]["para"])
        assert para["id"] == "1"
        assert para["name"] == "New Name"

    def test_set_holiday(self, mock_session: Any) -> None:
        """Test setting holiday mode."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }
        mock_session.put.return_value = mock_response

        holiday = Holiday(
            enable=True,
            start=datetime(2025, 12, 20, 0, 0, 0),
            end=datetime(2026, 1, 5, 23, 59, 59)
        )

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        client.set_holiday(holiday)

        call_args = mock_session.put.call_args
        holiday_data = json.loads(call_args[1]["data"]["holiday"])
        assert holiday_data["enable"] == "1"
        assert holiday_data["start"] == "2025-12-20 00:00:00"

    def test_get_holiday(self, mock_session: Any) -> None:
        """Test getting holiday configuration."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": True,
            "message": {
                "enable": "1",
                "start": "2025-08-01 00:00:00",
                "end": "2025-08-31 23:59:59"
            },
            "error_code": 0
        }
        mock_session.get.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        holiday = client.get_holiday()

        assert holiday.enable is True
        assert holiday.start == datetime(2025, 8, 1, 0, 0, 0)
        assert holiday.end == datetime(2025, 8, 31, 23, 59, 59)

    def test_api_error_response(self, mock_session: Any) -> None:
        """Test handling of API error responses."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": False,
            "message": "Device not found",
            "error_code": 404
        }
        mock_session.get.return_value = mock_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        with pytest.raises(SupplyMasterError, match="Failed to list devices"):
            client.list_devices()

    def test_network_error(self, mock_session: Any) -> None:
        """Test handling of network errors."""
        mock_session.get.side_effect = requests.RequestException("Timeout")

        client = SupplyMasterClient(user_id="123", token="abc", device_id="device1")
        with pytest.raises(SupplyMasterError, match="Failed to list devices"):
            client.list_devices()

    def test_custom_timeout(self, mock_session: Any) -> None:
        """Test custom timeout setting."""
        client = SupplyMasterClient(user_id="123", token="abc", timeout=60)
        assert client.timeout == 60

    def test_user_agent_header(self, mock_session: Any) -> None:
        """Test that User-Agent header is set correctly."""
        client = SupplyMasterClient(user_id="123", token="abc")
        mock_session.headers.update.assert_called_with({"User-Agent": "okhttp/3.3.1"})


class TestIntegration:
    """Integration-style tests with multiple operations."""

    @pytest.fixture
    def mock_session(self) -> Generator[Any, None, None]:
        """Create a mock session for integration tests."""
        with patch('nest_octopus.tg_supplymaster.requests.Session') as mock:
            session = Mock()
            mock.return_value = session
            yield session

    def test_full_workflow(self, mock_session: Any) -> None:
        """Test a complete workflow: authenticate, list devices, control device."""
        # Mock authentication
        auth_response = Mock()
        auth_response.json.return_value = {
            "status": True,
            "message": {
                "user": {"id": "123", "token": "token123"},
                "wifi_box": [{"device_id": "device1", "name": "My Device"}]
            },
            "error_code": 0
        }

        # Mock list devices
        list_response = Mock()
        list_response.json.return_value = {
            "status": True,
            "message": [
                {
                    "device_id": "device123",
                    "name": "Living Room Switch",
                    "online": "1",
                    "relay": "0",
                    "work_mode": 1,
                    "advance": "0",
                    "main_relay": {"loaded": "1", "work_status": "0"}
                }
            ],
            "error_code": 0
        }

        # Mock set mode
        mode_response = Mock()
        mode_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }

        mock_session.put.return_value = auth_response
        mock_session.get.return_value = list_response

        # Authentication
        client = SupplyMasterClient(username="user", password="pass")
        assert client.device_id == "device1"

        # List devices
        mock_session.get.return_value = list_response
        devices = client.list_devices()
        assert len(devices) == 1
        assert devices[0].work_mode == WorkMode.ALWAYS_OFF

        # Turn on
        mock_session.put.return_value = mode_response
        client.set_mode(WorkMode.ALWAYS_ON)

        # Verify call was made
        assert mock_session.put.call_count >= 2  # auth + set_mode

    def test_program_management(self, mock_session: Any) -> None:
        """Test creating and enabling a program."""
        # Mock get program
        get_response = Mock()
        get_response.json.return_value = {
            "status": True,
            "message": {
                "id": "0",
                "name": "Default",
                "program": [
                    {
                        "start": {"enable": "0", "time": "00:00"},
                        "end": {"enable": "0", "time": "00:00"},
                        "map": {str(i): "0" for i in range(7)}
                    }
                ] * 6  # 6 empty slots
            },
            "error_code": 0
        }

        # Mock update/enable responses
        success_response = Mock()
        success_response.json.return_value = {
            "status": True,
            "message": "Set successfully",
            "error_code": 0
        }

        mock_session.get.return_value = get_response
        mock_session.put.return_value = success_response

        client = SupplyMasterClient(user_id="123", token="abc", device_id="dev1")

        # Get existing program
        program = client.get_program("0")
        assert program.name == "Default"

        # Modify it
        program.name = "My Schedule"
        days_weekdays = {
            DayOfWeek.SUNDAY: False,
            DayOfWeek.MONDAY: True,
            DayOfWeek.TUESDAY: True,
            DayOfWeek.WEDNESDAY: True,
            DayOfWeek.THURSDAY: True,
            DayOfWeek.FRIDAY: True,
            DayOfWeek.SATURDAY: False,
        }
        program.slots[0] = ProgramSlot(
            start=TimeSlot(True, "07:00"),
            end=TimeSlot(True, "09:00"),
            days=days_weekdays
        )

        # Update program
        client.update_program(program)

        # Enable it
        client.enable_program("0")

        # Verify calls
        assert mock_session.put.call_count == 2
