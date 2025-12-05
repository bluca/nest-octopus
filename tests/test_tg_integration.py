# SPDX-License-Identifier: MPL-2.0
"""Tests for TG SupplyMaster integration in heating optimizer."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from typing import Any, Generator

import pytest

from nest_octopus.heating_optimizer import find_cheapest_windows, program_tg_switch, Config
from nest_octopus.octopus import PricePoint


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
