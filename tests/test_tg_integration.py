# SPDX-License-Identifier: MPL-2.0
"""Tests for TG SupplyMaster integration in heating optimizer."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from typing import Any, Generator

import pytest

from nest_octopus.heating_optimizer import (
    find_cheapest_windows,
    program_tg_switch,
    Config,
    _existing_program_has_pending_slots,
)
from nest_octopus.octopus import PricePoint
from nest_octopus.tg_supplymaster import (
    DayOfWeek,
    Program,
    ProgramSlot,
    TimeSlot,
)


class TestFindCheapestWindows:
    """Tests for find_cheapest_windows function."""

    def test_find_two_cheapest_windows(self) -> None:
        """Test finding two cheapest 2-hour windows."""
        # Create 24 hours of price data (48 half-hour slots)
        base_time = datetime(2025, 12, 5, 0, 0, 0)
        prices = []

        # Create price pattern with two cheap windows
        for i in range(48):
            time = base_time + timedelta(minutes=30 * i)

            # Cheap window 1: 02:00-04:00 (slots 4-7)
            if 4 <= i < 8:
                price = 10.0
            # Cheap window 2: 14:00-16:00 (slots 28-31)
            elif 28 <= i < 32:
                price = 11.0
            # Expensive otherwise
            else:
                price = 25.0

            prices.append(PricePoint({
                'value_inc_vat': price,
                'value_exc_vat': price / 1.05,
                'valid_from': time.isoformat(),
                'valid_to': (time + timedelta(minutes=30)).isoformat()
            }))

        windows = find_cheapest_windows(prices, window_hours=2, num_windows=2, min_gap_hours=10)

        assert len(windows) == 2

        # First window should be 02:00-04:00
        assert windows[0][0].hour == 2
        assert windows[0][0].minute == 0
        assert windows[0][1].hour == 4
        assert windows[0][1].minute == 0

        # Second window should be 14:00-16:00
        assert windows[1][0].hour == 14
        assert windows[1][0].minute == 0
        assert windows[1][1].hour == 16
        assert windows[1][1].minute == 0

    def test_respects_minimum_gap(self) -> None:
        """Test that windows respect minimum gap constraint."""
        base_time = datetime(2025, 12, 5, 0, 0, 0)
        prices = []

        # Create price pattern with three cheap windows but only two far enough apart
        for i in range(48):
            time = base_time + timedelta(minutes=30 * i)

            # Cheap window 1: 02:00-04:00
            if 4 <= i < 8:
                price = 10.0
            # Cheap window 2: 08:00-10:00 (only 4 hours gap - should be rejected)
            elif 16 <= i < 20:
                price = 11.0
            # Cheap window 3: 20:00-22:00 (12 hours from window 1 - should be selected)
            elif 40 <= i < 44:
                price = 12.0
            else:
                price = 25.0

            prices.append(PricePoint({
                'value_inc_vat': price,
                'value_exc_vat': price / 1.05,
                'valid_from': time.isoformat(),
                'valid_to': (time + timedelta(minutes=30)).isoformat()
            }))

        windows = find_cheapest_windows(prices, window_hours=2, num_windows=2, min_gap_hours=10)

        assert len(windows) == 2

        # Should select window 1 (02:00) and window 3 (20:00), skipping window 2 (08:00)
        assert windows[0][0].hour == 2
        assert windows[1][0].hour == 20

    def test_insufficient_data(self) -> None:
        """Test handling of insufficient price data."""
        base_time = datetime(2025, 12, 5, 0, 0, 0)

        # Only 2 price points (not enough for a 2-hour window)
        prices = [
            PricePoint({
                'value_inc_vat': 10.0,
                'value_exc_vat': 9.5,
                'valid_from': base_time.isoformat(),
                'valid_to': (base_time + timedelta(minutes=30)).isoformat()
            }),
            PricePoint({
                'value_inc_vat': 11.0,
                'value_exc_vat': 10.5,
                'valid_from': (base_time + timedelta(minutes=30)).isoformat(),
                'valid_to': (base_time + timedelta(hours=1)).isoformat()
            })
        ]

        windows = find_cheapest_windows(prices, window_hours=2, num_windows=2, min_gap_hours=10)

        assert len(windows) == 0

    def test_custom_window_size(self) -> None:
        """Test with different window size."""
        base_time = datetime(2025, 12, 5, 0, 0, 0)
        prices = []

        # Create 24 hours of price data
        for i in range(48):
            time = base_time + timedelta(minutes=30 * i)

            # Cheap window: 02:00-05:00 (3 hours = 6 slots)
            if 4 <= i < 10:
                price = 10.0
            else:
                price = 25.0

            prices.append(PricePoint({
                'value_inc_vat': price,
                'value_exc_vat': price / 1.05,
                'valid_from': time.isoformat(),
                'valid_to': (time + timedelta(minutes=30)).isoformat()
            }))

        # Find 3-hour window
        windows = find_cheapest_windows(prices, window_hours=3, num_windows=1, min_gap_hours=10)

        assert len(windows) == 1
        assert windows[0][0].hour == 2
        assert windows[0][1].hour == 5


class TestProgramTgSwitch:
    """Tests for program_tg_switch function."""

    @pytest.fixture
    def mock_tg_client(self) -> Generator[Any, None, None]:
        """Create a mock TG SupplyMasterClient."""
        with patch('nest_octopus.heating_optimizer.SupplyMasterClient') as mock_class:
            client_instance = Mock()
            client_instance.__enter__ = Mock(return_value=client_instance)
            client_instance.__exit__ = Mock(return_value=False)
            mock_class.return_value = client_instance
            # Yield both the class mock and the instance for convenience
            yield mock_class

    def test_skip_if_not_configured(self, mock_tg_client: Any) -> None:
        """Test that TG programming is skipped if not configured."""
        config = Config(
            thermostat_name="Test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="project",
            tg_username=None,  # Not configured
            tg_password=None
        )

        windows = [(datetime.now(), datetime.now() + timedelta(hours=2), 10.0)]

        program_tg_switch(config, windows)

        # Should not create client
        mock_tg_client.assert_not_called()

    def test_program_with_two_windows(self, mock_tg_client: Any) -> None:
        """Test programming switch with two windows."""
        config = Config(
            thermostat_name="Test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="project",
            tg_username="tg_user",
            tg_password="test_password"
        )

        mock_tg_client.return_value.list_programs.return_value = {
            'choosex': '0',
            'namelist': [
                {'id': '0', 'name': 'Default Program'},
                {'id': '1', 'name': ''},  # Unused slot
                {'id': '2', 'name': 'Some Other Program'}
            ]
        }

        start1 = datetime(2025, 12, 5, 2, 0, 0)
        end1 = datetime(2025, 12, 5, 4, 0, 0)
        start2 = datetime(2025, 12, 5, 14, 0, 0)
        end2 = datetime(2025, 12, 5, 16, 0, 0)

        windows = [
            (start1, end1, 10.5),
            (start2, end2, 11.2)
        ]

        program_tg_switch(config, windows)

        # Verify client was created with correct credentials
        mock_tg_client.assert_called_once()
        call_kwargs = mock_tg_client.call_args[1]
        assert call_kwargs['username'] == 'tg_user'
        assert call_kwargs['password'] == 'test_password'

        # Verify program was updated
        mock_tg_client.return_value.update_program.assert_called_once()
        program = mock_tg_client.return_value.update_program.call_args[0][0]
        assert program.name == "Agile Optimized"
        assert len(program.slots) == 6  # Total slots including empty ones

        # Verify first two slots have correct times
        assert program.slots[0].start.time == "02:00"
        assert program.slots[0].end.time == "04:00"
        assert program.slots[0].start.enable is True

        assert program.slots[1].start.time == "14:00"
        assert program.slots[1].end.time == "16:00"
        assert program.slots[1].start.enable is True

        # Verify remaining slots are empty
        for i in range(2, 6):
            assert program.slots[i].start.enable is False

        # Verify program was enabled (but mode NOT changed)
        mock_tg_client.return_value.enable_program.assert_called_once_with("1")
        mock_tg_client.set_mode.assert_not_called()

    def test_select_device_by_name(self, mock_tg_client: Any) -> None:
        """Test device selection by name."""
        config = Config(
            thermostat_name="Test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="project",
            tg_username="tg_user",
            tg_password="password",
            tg_device_name="Kitchen Switch"
        )

        # Mock device list
        from nest_octopus.tg_supplymaster import DeviceInfo, WorkMode
        mock_devices = [
            DeviceInfo(
                device_id="dev1",
                name="Kitchen Switch",
                online=True,
                relay=False,
                work_mode=WorkMode.AUTO,
                advance=False,
                loaded=True,
                work_status=False
            ),
            DeviceInfo(
                device_id="dev2",
                name="Bedroom Switch",
                online=True,
                relay=False,
                work_mode=WorkMode.AUTO,
                advance=False,
                loaded=True,
                work_status=False
            )
        ]
        mock_tg_client.return_value.list_devices.return_value = mock_devices
        mock_tg_client.return_value.list_programs.return_value = {
            'choosex': '0',
            'namelist': [{'id': '0', 'name': ''}]
        }

        windows = [(datetime.now(), datetime.now() + timedelta(hours=2), 10.0)]

        program_tg_switch(config, windows)

        # Verify correct device was selected
        assert mock_tg_client.return_value.device_id == "dev1"

    def test_handles_errors_gracefully(self, mock_tg_client: Any) -> None:
        """Test that errors don't crash the main cycle."""
        config = Config(
            thermostat_name="Test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="project",
            tg_username="tg_user",
            tg_password="password"
        )

        # Make client raise an error
        mock_tg_client.return_value.list_programs.side_effect = Exception("Network error")

        windows = [(datetime.now(), datetime.now() + timedelta(hours=2), 10.0)]

        # Should not raise - errors are caught and logged
        program_tg_switch(config, windows)

    def test_reuses_existing_program(self, mock_tg_client: Any) -> None:
        """Test that existing program with same name is reused."""
        config = Config(
            thermostat_name="Test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="project",
            tg_username="tg_user",
            tg_password="password"
        )

        mock_tg_client.return_value.list_programs.return_value = {
            'choosex': '0',
            'namelist': [
                {'id': '0', 'name': 'Default'},
                {'id': '1', 'name': 'Agile Optimized'},  # Existing program with same name
                {'id': '2', 'name': ''}
            ]
        }

        windows = [(datetime.now(), datetime.now() + timedelta(hours=2), 10.0)]

        program_tg_switch(config, windows)

        # Should reuse existing program ID 1
        mock_tg_client.return_value.enable_program.assert_called_once_with("1")

    def test_uses_unused_slot(self, mock_tg_client: Any) -> None:
        """Test that unused program slot (empty name) is used for new program."""
        config = Config(
            thermostat_name="Test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="project",
            tg_username="tg_user",
            tg_password="password"
        )

        mock_tg_client.return_value.list_programs.return_value = {
            'choosex': '0',
            'namelist': [
                {'id': '0', 'name': 'Default'},
                {'id': '1', 'name': 'Other Program'},
                {'id': '2', 'name': ''},  # First unused slot
                {'id': '3', 'name': ''}
            ]
        }

        windows = [(datetime.now(), datetime.now() + timedelta(hours=2), 10.0)]

        program_tg_switch(config, windows)

        # Should use first unused slot (ID 2)
        mock_tg_client.return_value.enable_program.assert_called_once_with("2")


class TestExistingProgramHasPendingSlots:
    """Tests for _existing_program_has_pending_slots function."""

    def test_no_pending_slots_all_past(self) -> None:
        """Test that all slots in the past returns False."""
        # Current time is 15:00, slots ended at 08:00 and 12:00
        now_local = datetime(2025, 12, 8, 15, 0, 0)  # Monday

        all_days = {day: True for day in DayOfWeek}
        program = Program(
            id="0",
            name="Agile Optimized",
            slots=[
                ProgramSlot(
                    start=TimeSlot(enable=True, time="06:00"),
                    end=TimeSlot(enable=True, time="08:00"),
                    days=all_days
                ),
                ProgramSlot(
                    start=TimeSlot(enable=True, time="10:00"),
                    end=TimeSlot(enable=True, time="12:00"),
                    days=all_days
                ),
                # Empty slots
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
            ]
        )

        mock_client = Mock()
        mock_client.get_program.return_value = program

        result = _existing_program_has_pending_slots(mock_client, "0", now_local)
        assert result is False

    def test_pending_slots_future_end_time(self) -> None:
        """Test that slot with future end time returns True."""
        # Current time is 15:00, one slot ends at 18:00
        now_local = datetime(2025, 12, 8, 15, 0, 0)  # Monday

        all_days = {day: True for day in DayOfWeek}
        program = Program(
            id="0",
            name="Agile Optimized",
            slots=[
                ProgramSlot(
                    start=TimeSlot(enable=True, time="06:00"),
                    end=TimeSlot(enable=True, time="08:00"),
                    days=all_days
                ),
                ProgramSlot(
                    start=TimeSlot(enable=True, time="16:00"),
                    end=TimeSlot(enable=True, time="18:00"),  # Hasn't ended yet
                    days=all_days
                ),
                # Empty slots
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
            ]
        )

        mock_client = Mock()
        mock_client.get_program.return_value = program

        result = _existing_program_has_pending_slots(mock_client, "0", now_local)
        assert result is True

    def test_slot_not_active_today(self) -> None:
        """Test that slot not active today is ignored."""
        # Current time is 15:00 on Monday, but slot is only for Saturday
        now_local = datetime(2025, 12, 8, 15, 0, 0)  # Monday

        saturday_only = {day: day == DayOfWeek.SATURDAY for day in DayOfWeek}
        program = Program(
            id="0",
            name="Agile Optimized",
            slots=[
                ProgramSlot(
                    start=TimeSlot(enable=True, time="16:00"),
                    end=TimeSlot(enable=True, time="18:00"),  # Future time but wrong day
                    days=saturday_only
                ),
                # Empty slots
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
            ]
        )

        mock_client = Mock()
        mock_client.get_program.return_value = program

        result = _existing_program_has_pending_slots(mock_client, "0", now_local)
        assert result is False

    def test_api_error_returns_false(self) -> None:
        """Test that API error returns False (to allow update)."""
        now_local = datetime(2025, 12, 8, 15, 0, 0)

        mock_client = Mock()
        mock_client.get_program.side_effect = Exception("API Error")

        result = _existing_program_has_pending_slots(mock_client, "0", now_local)
        assert result is False


class TestProgramTGSwitchPendingSlots:
    """Tests for program_tg_switch with pending slots logic."""

    @pytest.fixture
    def mock_tg_client(self) -> Generator[Any, None, None]:
        """Create mock TG SupplyMaster client."""
        with patch('nest_octopus.heating_optimizer.SupplyMasterClient') as mock:
            mock.return_value.__enter__ = Mock(return_value=mock.return_value)
            mock.return_value.__exit__ = Mock(return_value=False)
            mock.return_value.device_id = "test_device"
            yield mock

    def test_skips_update_when_pending_slots_exist(self, mock_tg_client: Any) -> None:
        """Test that update is skipped when existing program has pending slots."""
        config = Config(
            thermostat_name="Test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="project",
            tg_username="tg_user",
            tg_password="password"
        )

        mock_tg_client.return_value.list_programs.return_value = {
            'choosex': '0',
            'namelist': [
                {'id': '0', 'name': 'Default'},
                {'id': '1', 'name': 'Agile Optimized'},  # Existing program
            ]
        }

        # Mock the get_program to return a program with pending slots
        all_days = {day: True for day in DayOfWeek}
        existing_program = Program(
            id="1",
            name="Agile Optimized",
            slots=[
                ProgramSlot(
                    start=TimeSlot(enable=True, time="23:00"),
                    end=TimeSlot(enable=True, time="23:59"),  # Future time
                    days=all_days
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
            ]
        )
        mock_tg_client.return_value.get_program.return_value = existing_program

        windows = [(datetime.now(), datetime.now() + timedelta(hours=2), 10.0)]

        program_tg_switch(config, windows)

        # update_program should NOT be called because of pending slots
        mock_tg_client.return_value.update_program.assert_not_called()
        mock_tg_client.return_value.enable_program.assert_not_called()

    def test_updates_when_all_slots_passed(self, mock_tg_client: Any) -> None:
        """Test that update proceeds when all existing slots have passed."""
        config = Config(
            thermostat_name="Test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="project",
            tg_username="tg_user",
            tg_password="password"
        )

        mock_tg_client.return_value.list_programs.return_value = {
            'choosex': '0',
            'namelist': [
                {'id': '0', 'name': 'Default'},
                {'id': '1', 'name': 'Agile Optimized'},  # Existing program
            ]
        }

        # Mock the get_program to return a program with all slots in the past
        all_days = {day: True for day in DayOfWeek}
        existing_program = Program(
            id="1",
            name="Agile Optimized",
            slots=[
                ProgramSlot(
                    start=TimeSlot(enable=True, time="02:00"),
                    end=TimeSlot(enable=True, time="04:00"),  # Past time
                    days=all_days
                ),
                ProgramSlot(
                    start=TimeSlot(enable=True, time="06:00"),
                    end=TimeSlot(enable=True, time="08:00"),  # Past time
                    days=all_days
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
                ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days={day: False for day in DayOfWeek}
                ),
            ]
        )
        mock_tg_client.return_value.get_program.return_value = existing_program

        windows = [(datetime.now(), datetime.now() + timedelta(hours=2), 10.0)]

        program_tg_switch(config, windows)

        # update_program SHOULD be called since all slots have passed
        mock_tg_client.return_value.update_program.assert_called_once()
        mock_tg_client.return_value.enable_program.assert_called_once_with("1")

    def test_updates_new_program_without_checking(self, mock_tg_client: Any) -> None:
        """Test that new program (no existing Agile Optimized) is always created."""
        config = Config(
            thermostat_name="Test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="project",
            tg_username="tg_user",
            tg_password="password"
        )

        mock_tg_client.return_value.list_programs.return_value = {
            'choosex': '0',
            'namelist': [
                {'id': '0', 'name': 'Default'},
                {'id': '1', 'name': ''},  # Unused slot
            ]
        }

        windows = [(datetime.now(), datetime.now() + timedelta(hours=2), 10.0)]

        program_tg_switch(config, windows)

        # get_program should NOT be called since no existing Agile Optimized
        mock_tg_client.return_value.get_program.assert_not_called()

        # update_program SHOULD be called
        mock_tg_client.return_value.update_program.assert_called_once()
        mock_tg_client.return_value.enable_program.assert_called_once_with("1")
