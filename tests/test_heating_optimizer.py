# SPDX-License-Identifier: MPL-2.0
"""
Comprehensive unit tests for the heating optimization daemon.

Tests configuration loading, price analysis, scheduling algorithm,
and thermostat control with mocked REST endpoints and time functions.
"""

import json
import os
import signal
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from nest_octopus.heating_optimizer import (
    Config,
    ConfigurationError,
    HeatingAction,
    PriceCategory,
    calculate_heating_schedule,
    calculate_price_statistics,
    classify_price,
    execute_heating_action,
    find_default_config,
    find_thermostat,
    handle_reload_signal,
    handle_shutdown_signal,
    load_config,
    run_daily_cycle,
    run_dry_run,
)
from nest_octopus.octopus import PricePoint


# Fixture directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "heating"


def create_price_point(valid_from_str, valid_to_str, price_inc_vat):
    """Helper to create a PricePoint for testing."""
    return PricePoint({
        'valid_from': valid_from_str,
        'valid_to': valid_to_str,
        'value_exc_vat': price_inc_vat / 1.05,
        'value_inc_vat': price_inc_vat
    })


def load_price_fixture(filename: str) -> list:
    """Load price fixture and convert to PricePoint objects."""
    with open(FIXTURES_DIR / filename) as f:
        data = json.load(f)

    prices = []
    for item in data['results']:
        price_obj = PricePoint(item)
        prices.append(price_obj)

    return prices


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_load_config_success(self, tmp_path):
        """Test successful configuration loading."""
        # Create config file
        config_file = tmp_path / "config.ini"
        config_file.write_text("""
[octopus]
tariff_code = E-1R-AGILE-FLEX-22-11-25-H

[nest]
thermostat_name = Living Room
client_id = test-client-id.apps.googleusercontent.com
project_id = test-project-123

[heating]
low_price_temp = 22.0
average_price_temp = 17.0
peak_start_hour = 16
peak_end_hour = 19
""")

        # Create credentials directory
        creds_dir = tmp_path / "credentials"
        creds_dir.mkdir()
        (creds_dir / "client_secret").write_text("test-secret-abc123")
        (creds_dir / "refresh_token").write_text("test-refresh-xyz789")

        # Set environment variable
        with patch.dict(os.environ, {'CREDENTIALS_DIRECTORY': str(creds_dir)}):
            config = load_config(str(config_file))

        assert config.tariff_code == "E-1R-AGILE-FLEX-22-11-25-H"
        assert config.thermostat_name == "Living Room"
        assert config.client_id == "test-client-id.apps.googleusercontent.com"
        assert config.client_secret == "test-secret-abc123"
        assert config.refresh_token == "test-refresh-xyz789"
        assert config.project_id == "test-project-123"
        assert config.low_price_temp == 22.0
        assert config.average_price_temp == 17.0
        assert config.peak_start_hour == 16
        assert config.peak_end_hour == 19

    def test_load_config_missing_file(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config("nonexistent.ini")

    def test_load_config_no_credentials_dir(self, tmp_path):
        """Test error when CREDENTIALS_DIRECTORY not set."""
        config_file = tmp_path / "config.ini"
        config_file.write_text("[octopus]\ntariff_code = TEST\n")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="CREDENTIALS_DIRECTORY"):
                load_config(str(config_file))

    def test_load_config_missing_credentials(self, tmp_path):
        """Test error when credential files are missing."""
        config_file = tmp_path / "config.ini"
        config_file.write_text("""
[octopus]
tariff_code = TEST
[nest]
thermostat_name = Test
client_id = test-id
project_id = test-proj
""")

        creds_dir = tmp_path / "credentials"
        creds_dir.mkdir()

        with patch.dict(os.environ, {'CREDENTIALS_DIRECTORY': str(creds_dir)}):
            with pytest.raises(ConfigurationError, match="client_secret"):
                load_config(str(config_file))

    def test_load_config_default_heating_values(self, tmp_path):
        """Test default heating configuration values."""
        config_file = tmp_path / "config.ini"
        config_file.write_text("""
[octopus]
tariff_code = TEST

[nest]
thermostat_name = Test
client_id = test-id
project_id = test-proj
""")

        creds_dir = tmp_path / "credentials"
        creds_dir.mkdir()
        (creds_dir / "client_secret").write_text("secret")
        (creds_dir / "refresh_token").write_text("token")

        with patch.dict(os.environ, {'CREDENTIALS_DIRECTORY': str(creds_dir)}):
            config = load_config(str(config_file))

        # Check default values
        assert config.low_price_temp == 22.0
        assert config.average_price_temp == 17.0
        assert config.peak_start_hour == 16
        assert config.peak_end_hour == 19

    def test_find_default_config_not_found(self):
        """Test find_default_config returns None when no files exist."""
        with patch('os.path.exists', return_value=False):
            result = find_default_config()
            assert result is None

    def test_find_default_config_finds_etc(self):
        """Test find_default_config finds /etc config first."""
        def exists_mock(path):
            return path == "/etc/nest-octopus/nest-octopus.conf"

        with patch('os.path.exists', side_effect=exists_mock):
            result = find_default_config()
            assert result == "/etc/nest-octopus/nest-octopus.conf"

    def test_find_default_config_finds_run(self):
        """Test find_default_config finds /run config if /etc missing."""
        def exists_mock(path):
            return path == "/run/nest-octopus/nest-octopus.conf"

        with patch('os.path.exists', side_effect=exists_mock):
            result = find_default_config()
            assert result == "/run/nest-octopus/nest-octopus.conf"

    def test_find_default_config_finds_usr_lib(self):
        """Test find_default_config finds /usr/lib config if others missing."""
        def exists_mock(path):
            return path == "/usr/lib/nest-octopus/nest-octopus.conf"

        with patch('os.path.exists', side_effect=exists_mock):
            result = find_default_config()
            assert result == "/usr/lib/nest-octopus/nest-octopus.conf"

    def test_load_config_with_none_path_searches(self, tmp_path):
        """Test load_config searches default paths when None provided."""
        # Create a config in a mock location
        config_file = tmp_path / "config.ini"
        config_file.write_text("""
[octopus]
tariff_code = TEST
[nest]
thermostat_name = Test
client_id = test-id
project_id = test-proj
""")

        creds_dir = tmp_path / "credentials"
        creds_dir.mkdir()
        (creds_dir / "client_secret").write_text("secret")
        (creds_dir / "refresh_token").write_text("token")

        with patch.dict(os.environ, {'CREDENTIALS_DIRECTORY': str(creds_dir)}):
            with patch('nest_octopus.heating_optimizer.find_default_config', return_value=str(config_file)):
                config = load_config(None)
                assert config.tariff_code == "TEST"

    def test_load_config_with_none_path_fails_if_not_found(self):
        """Test load_config raises error when no default config found."""
        with patch('nest_octopus.heating_optimizer.find_default_config', return_value=None):
            with pytest.raises(ConfigurationError, match="No configuration file found"):
                load_config(None)


class TestPriceAnalysis:
    """Test price statistics and classification."""

    def test_calculate_price_statistics(self):
        """Test price statistics calculation."""
        daily_prices = load_price_fixture("typical_day_prices.json")
        weekly_prices = load_price_fixture("weekly_prices_sample.json")

        daily_avg, weekly_avg, daily_min, daily_max = calculate_price_statistics(
            daily_prices, weekly_prices
        )

        # Check reasonable values
        assert daily_min < daily_avg < daily_max
        assert 0 < daily_avg < 50
        assert 0 < weekly_avg < 50
        assert daily_min >= 0

    def test_classify_price_peak_hours(self):
        """Test that prices during 16:00-19:00 are classified as PEAK."""
        price = create_price_point(
            '2024-12-02T17:00:00Z',
            '2024-12-02T17:30:00Z',
            45.5
        )

        category = classify_price(price, daily_avg=15.0, weekly_avg=14.0,
                                 peak_start_hour=16, peak_end_hour=19)

        assert category == PriceCategory.PEAK

    def test_classify_price_low(self):
        """Test classification of low prices."""
        price = create_price_point(
            '2024-12-02T02:00:00Z',
            '2024-12-02T02:30:00Z',
            5.3
        )

        category = classify_price(price, daily_avg=15.0, weekly_avg=14.0,
                                 peak_start_hour=16, peak_end_hour=19)

        assert category == PriceCategory.LOW

    def test_classify_price_average(self):
        """Test classification of average prices."""
        price = create_price_point(
            '2024-12-02T10:00:00Z',
            '2024-12-02T10:30:00Z',
            14.5  # Between avg and high threshold
        )

        category = classify_price(price, daily_avg=15.0, weekly_avg=14.0,
                                 peak_start_hour=16, peak_end_hour=19)

        assert category == PriceCategory.AVERAGE

    def test_classify_price_high(self):
        """Test classification of high prices."""
        price = create_price_point(
            '2024-12-02T15:00:00Z',
            '2024-12-02T15:30:00Z',
            20.3
        )

        category = classify_price(price, daily_avg=15.0, weekly_avg=14.0,
                                 peak_start_hour=16, peak_end_hour=19)

        assert category == PriceCategory.HIGH


class TestHeatingSchedule:
    """Test heating schedule calculation."""

    def test_calculate_heating_schedule_typical_day(self):
        """Test schedule calculation for a typical day."""
        daily_prices = load_price_fixture("typical_day_prices.json")
        weekly_prices = load_price_fixture("weekly_prices_sample.json")

        config = Config(
            tariff_code="TEST",
            thermostat_name="Test",
            client_id="test-id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj",
            low_price_temp=22.0,
            average_price_temp=17.0,
            peak_start_hour=16,
            peak_end_hour=19
        )

        start_time = datetime(2024, 12, 1, 20, 0)
        actions = calculate_heating_schedule(daily_prices, weekly_prices, config, start_time)

        # Should have multiple actions
        assert len(actions) > 0

        # Actions should be sorted by time
        for i in range(len(actions) - 1):
            assert actions[i].timestamp <= actions[i + 1].timestamp

        # Should have ECO mode during peak hours (16:00-19:00)
        peak_actions = [a for a in actions if 16 <= a.timestamp.hour < 19]
        if peak_actions:
            assert any(a.eco_mode for a in peak_actions), "Should enable ECO during peak hours"

        # Should have low temperature setpoint during low price periods
        low_temp_actions = [a for a in actions if a.temperature == 22.0]
        assert len(low_temp_actions) > 0, "Should heat to 22°C during low prices"

    def test_heating_action_representation(self):
        """Test HeatingAction string representation."""
        eco_action = HeatingAction(
            timestamp=datetime(2024, 12, 2, 17, 0),
            temperature=None,
            eco_mode=True,
            reason="Peak price"
        )

        temp_action = HeatingAction(
            timestamp=datetime(2024, 12, 2, 2, 0),
            temperature=22.0,
            eco_mode=False,
            reason="Low price"
        )

        assert "ECO" in repr(eco_action)
        assert "22" in repr(temp_action)


class TestThermostatControl:
    """Test thermostat control functions."""

    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    def test_execute_heating_action_eco_mode(self, mock_client_class):
        """Test executing ECO mode action."""
        from nest_octopus.nest_thermostat import EcoMode, ThermostatStatus

        mock_client = Mock()
        # Mock get_device to return status with HEAT mode
        device_data = {
            'name': 'device-123',
            'traits': {
                'sdm.devices.traits.Temperature': {'ambientTemperatureCelsius': 20.0},
                'sdm.devices.traits.Humidity': {'ambientHumidityPercent': 50.0},
                'sdm.devices.traits.ThermostatMode': {'mode': 'HEAT'},
                'sdm.devices.traits.ThermostatEco': {'mode': 'OFF'},
                'sdm.devices.traits.ThermostatHvac': {'status': 'OFF'},
                'sdm.devices.traits.ThermostatTemperatureSetpoint': {'heatCelsius': 20.0}
            }
        }
        mock_status = ThermostatStatus(device_data)
        mock_client.get_device.return_value = mock_status

        action = HeatingAction(
            timestamp=datetime(2024, 12, 2, 17, 0),
            temperature=None,
            eco_mode=True,
            reason="Peak hours"
        )

        execute_heating_action(action, mock_client, "device-123")

        # Should call set_eco_mode with MANUAL_ECO
        mock_client.set_eco_mode.assert_called_once()
        call_args = mock_client.set_eco_mode.call_args
        assert call_args[0][0] == "device-123"
        assert call_args[0][1] == EcoMode.MANUAL_ECO

    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    def test_execute_heating_action_temperature(self, mock_client_class):
        """Test executing temperature setpoint action."""
        from nest_octopus.nest_thermostat import ThermostatStatus

        mock_client = Mock()
        # Mock get_device to return status with HEAT mode
        device_data = {
            'name': 'device-123',
            'traits': {
                'sdm.devices.traits.Temperature': {'ambientTemperatureCelsius': 20.0},
                'sdm.devices.traits.Humidity': {'ambientHumidityPercent': 50.0},
                'sdm.devices.traits.ThermostatMode': {'mode': 'HEAT'},
                'sdm.devices.traits.ThermostatEco': {'mode': 'OFF'},
                'sdm.devices.traits.ThermostatHvac': {'status': 'OFF'},
                'sdm.devices.traits.ThermostatTemperatureSetpoint': {'heatCelsius': 20.0}
            }
        }
        mock_status = ThermostatStatus(device_data)
        mock_client.get_device.return_value = mock_status

        action = HeatingAction(
            timestamp=datetime(2024, 12, 2, 2, 0),
            temperature=22.0,
            eco_mode=False,
            reason="Low price"
        )

        execute_heating_action(action, mock_client, "device-123")

        # Should disable ECO, set HEAT mode, then set temperature
        assert mock_client.set_eco_mode.call_count >= 1
        mock_client.set_heat.assert_called_once_with("device-123", 22.0)

    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    def test_find_thermostat_by_name(self, mock_client_class):
        """Test finding thermostat by name."""
        mock_client = Mock()

        # Create mock devices
        mock_device1 = Mock()
        mock_device1.device_id = "enterprises/proj/devices/living-room-123"
        mock_device2 = Mock()
        mock_device2.device_id = "enterprises/proj/devices/bedroom-456"

        mock_client.list_devices.return_value = [mock_device1, mock_device2]

        device_id = find_thermostat(mock_client, "living-room")

        assert device_id == "enterprises/proj/devices/living-room-123"

    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    def test_find_thermostat_not_found(self, mock_client_class):
        """Test error when thermostat not found."""
        mock_client = Mock()
        mock_client.list_devices.return_value = []

        with pytest.raises(ValueError, match="not found"):
            find_thermostat(mock_client, "nonexistent")


class TestDailyCycle:
    """Test the daily optimization cycle."""

    @patch('nest_octopus.heating_optimizer.time.sleep')
    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    def test_run_daily_cycle_complete(self, mock_octopus_class,
                                     mock_nest_class, mock_sleep):
        """Test complete daily cycle execution."""
        from nest_octopus.nest_thermostat import ThermostatStatus

        # Mock Octopus client
        mock_octopus = Mock()
        mock_octopus_class.return_value = mock_octopus

        daily_prices = load_price_fixture("typical_day_prices.json")
        weekly_prices = load_price_fixture("weekly_prices_sample.json")

        mock_octopus.get_unit_rates.side_effect = [daily_prices, weekly_prices]

        # Mock Nest client
        mock_nest = Mock()
        mock_nest_class.return_value = mock_nest

        mock_device = Mock()
        mock_device.device_id = "enterprises/proj/devices/thermostat-123"
        mock_nest.list_devices.return_value = [mock_device]

        # Mock get_device to return status with HEAT mode
        device_data = {
            'name': 'enterprises/proj/devices/thermostat-123',
            'traits': {
                'sdm.devices.traits.Temperature': {'ambientTemperatureCelsius': 20.0},
                'sdm.devices.traits.Humidity': {'ambientHumidityPercent': 50.0},
                'sdm.devices.traits.ThermostatMode': {'mode': 'HEAT'},
                'sdm.devices.traits.ThermostatEco': {'mode': 'OFF'},
                'sdm.devices.traits.ThermostatHvac': {'status': 'OFF'},
                'sdm.devices.traits.ThermostatTemperatureSetpoint': {'heatCelsius': 20.0}
            }
        }
        mock_status = ThermostatStatus(device_data)
        mock_nest.get_device.return_value = mock_status

        # Create config
        config = Config(
            tariff_code="E-1R-AGILE-FLEX-22-11-25-H",
            thermostat_name="thermostat",
            client_id="test-id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj"
        )

        # Run cycle
        run_daily_cycle(config)

        # Verify Octopus API calls
        assert mock_octopus.get_unit_rates.call_count == 2

        # Verify get_unit_rates was called with correct parameters (tariff_code, period_from, period_to)
        for call in mock_octopus.get_unit_rates.call_args_list:
            args, kwargs = call
            # Should be called with keyword arguments only
            assert len(args) == 0, "get_unit_rates should be called with keyword arguments only"
            assert 'tariff_code' in kwargs, "tariff_code parameter is required"
            assert 'period_from' in kwargs, "period_from parameter is required"
            assert 'period_to' in kwargs, "period_to parameter is required"
            assert 'product_code' not in kwargs, "product_code should not be passed (it's derived internally)"
            # Verify tariff_code matches config
            assert kwargs['tariff_code'] == config.tariff_code
            # Verify period_from and period_to are ISO 8601 strings
            assert isinstance(kwargs['period_from'], str), "period_from should be a string"
            assert isinstance(kwargs['period_to'], str), "period_to should be a string"

        # Verify Nest API calls
        mock_nest.list_devices.assert_called_once()
        assert mock_nest.set_eco_mode.called or mock_nest.set_heat.called

        # Note: sleep may not be called if all actions are in the past
        # (which happens with static fixture data)

        # Verify clients were closed
        mock_octopus.close.assert_called_once()
        mock_nest.close.assert_called_once()

    @patch('nest_octopus.heating_optimizer.time.sleep')
    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    def test_run_daily_cycle_no_prices(self, mock_octopus_class,
                                       mock_nest_class, mock_sleep):
        """Test handling when no prices are available."""
        mock_octopus = Mock()
        mock_octopus_class.return_value = mock_octopus
        mock_octopus.get_unit_rates.return_value = []  # No prices

        mock_nest = Mock()
        mock_nest_class.return_value = mock_nest
        # Mock thermostat list even though we won't use it
        mock_device = Mock()
        mock_device.device_id = "test-device-id"
        mock_nest.list_devices.return_value = [mock_device]

        config = Config(
            tariff_code="TEST",
            thermostat_name="test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj"
        )

        run_daily_cycle(config)

        # Should not attempt to control thermostat
        assert not mock_nest.set_mode.called
        assert not mock_nest.set_heat.called

        # Should still close clients
        mock_octopus.close.assert_called_once()
        mock_nest.close.assert_called_once()


class TestSchedulingLogic:
    """Test specific scheduling scenarios."""

    def test_overnight_low_prices_heat_to_22(self):
        """Test that overnight low prices result in heating to 22°C."""
        # Create overnight low price scenario
        prices = [
            create_price_point('2024-12-02T02:00:00Z', '2024-12-02T02:30:00Z', 5.0),
            create_price_point('2024-12-02T02:30:00Z', '2024-12-02T03:00:00Z', 5.2)
        ]

        weekly_prices = [
            create_price_point('2024-11-25T02:00:00Z', '2024-11-25T02:30:00Z', 12.0)
        ]

        config = Config(
            tariff_code="TEST",
            thermostat_name="test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj",
            low_price_temp=22.0,
            average_price_temp=17.0
        )

        actions = calculate_heating_schedule(
            prices, weekly_prices, config, datetime(2024, 12, 1, 20, 0, tzinfo=timezone.utc)
        )

        # Should have action to heat to 22°C
        low_temp_actions = [a for a in actions if a.temperature == 22.0]
        assert len(low_temp_actions) > 0

    def test_peak_hours_always_eco(self):
        """Test that peak hours (16:00-19:00) always use ECO mode."""
        # Create peak price scenario
        prices = [
            create_price_point('2024-12-02T16:00:00Z', '2024-12-02T16:30:00Z', 35.0),
            create_price_point('2024-12-02T17:00:00Z', '2024-12-02T17:30:00Z', 45.0),
            create_price_point('2024-12-02T18:00:00Z', '2024-12-02T18:30:00Z', 40.0)
        ]

        weekly_prices = [
            create_price_point('2024-11-25T16:00:00Z', '2024-11-25T16:30:00Z', 15.0)
        ]

        config = Config(
            tariff_code="TEST",
            thermostat_name="test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj",
            peak_start_hour=16,
            peak_end_hour=19
        )

        actions = calculate_heating_schedule(
            prices, weekly_prices, config, datetime(2024, 12, 1, 20, 0, tzinfo=timezone.utc)
        )

        # All actions during peak hours should be ECO mode
        peak_actions = [a for a in actions if 16 <= a.timestamp.hour < 19]
        if peak_actions:
            assert all(a.eco_mode for a in peak_actions)

    def test_low_price_window_ends_returns_to_average_temp(self):
        """Test that temperature returns to average_price_temp when LOW price period ends."""
        # Create scenario: LOW prices 01:00-03:00, then AVERAGE prices 03:00-05:00
        prices = [
            create_price_point('2024-12-02T00:00:00Z', '2024-12-02T00:30:00Z', 20.0),  # AVERAGE
            create_price_point('2024-12-02T00:30:00Z', '2024-12-02T01:00:00Z', 20.0),  # AVERAGE
            create_price_point('2024-12-02T01:00:00Z', '2024-12-02T01:30:00Z', 8.0),   # LOW
            create_price_point('2024-12-02T01:30:00Z', '2024-12-02T02:00:00Z', 7.0),   # LOW
            create_price_point('2024-12-02T02:00:00Z', '2024-12-02T02:30:00Z', 9.0),   # LOW
            create_price_point('2024-12-02T02:30:00Z', '2024-12-02T03:00:00Z', 8.5),   # LOW
            create_price_point('2024-12-02T03:00:00Z', '2024-12-02T03:30:00Z', 18.0),  # AVERAGE
            create_price_point('2024-12-02T03:30:00Z', '2024-12-02T04:00:00Z', 19.0),  # AVERAGE
            create_price_point('2024-12-02T04:00:00Z', '2024-12-02T04:30:00Z', 20.0),  # AVERAGE
            create_price_point('2024-12-02T04:30:00Z', '2024-12-02T05:00:00Z', 18.5),  # AVERAGE
        ]

        weekly_prices = [
            create_price_point('2024-11-25T00:00:00Z', '2024-11-25T00:30:00Z', 15.0)
        ]

        config = Config(
            tariff_code="TEST",
            thermostat_name="test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj",
            low_price_temp=22.0,
            average_price_temp=17.0,
            peak_start_hour=16,
            peak_end_hour=19
        )

        actions = calculate_heating_schedule(
            prices, weekly_prices, config, datetime(2024, 12, 1, 20, 0, tzinfo=timezone.utc)
        )

        # Find the action that sets temperature to 22°C (entering LOW period)
        low_temp_action = next((a for a in actions if a.temperature == 22.0), None)
        assert low_temp_action is not None, "Should have action to heat to 22°C during LOW prices"

        # Find the action that returns to 17°C (exiting LOW period)
        return_to_average = [a for a in actions
                            if a.temperature == 17.0
                            and a.timestamp > low_temp_action.timestamp
                            and "End of LOW price period" in a.reason]
        assert len(return_to_average) > 0, "Should return to 17°C when LOW price period ends"

        # Verify the return happens at the expected time (03:00 when AVERAGE period starts)
        assert return_to_average[0].timestamp.hour == 3
        assert return_to_average[0].timestamp.minute == 0

    def test_low_price_to_peak_returns_to_average_then_eco(self):
        """Test that when LOW period transitions to PEAK, temp returns to average before ECO."""
        # Create scenario: LOW prices 14:00-16:00, then PEAK at 16:00
        prices = [
            create_price_point('2024-12-02T14:00:00Z', '2024-12-02T14:30:00Z', 8.0),   # LOW
            create_price_point('2024-12-02T14:30:00Z', '2024-12-02T15:00:00Z', 7.0),   # LOW
            create_price_point('2024-12-02T15:00:00Z', '2024-12-02T15:30:00Z', 9.0),   # LOW
            create_price_point('2024-12-02T15:30:00Z', '2024-12-02T16:00:00Z', 8.5),   # LOW
            create_price_point('2024-12-02T16:00:00Z', '2024-12-02T16:30:00Z', 35.0),  # PEAK
            create_price_point('2024-12-02T16:30:00Z', '2024-12-02T17:00:00Z', 40.0),  # PEAK
            create_price_point('2024-12-02T17:00:00Z', '2024-12-02T17:30:00Z', 38.0),  # PEAK
        ]

        weekly_prices = [
            create_price_point('2024-11-25T14:00:00Z', '2024-11-25T14:30:00Z', 15.0)
        ]

        config = Config(
            tariff_code="TEST",
            thermostat_name="test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj",
            low_price_temp=22.0,
            average_price_temp=17.0,
            peak_start_hour=16,
            peak_end_hour=19
        )

        actions = calculate_heating_schedule(
            prices, weekly_prices, config, datetime(2024, 12, 1, 20, 0, tzinfo=timezone.utc)
        )

        # Find the sequence of actions
        sorted_actions = sorted(actions, key=lambda a: a.timestamp)

        # Should have: 22°C (LOW), then 17°C (end of LOW), then ECO (PEAK)
        low_temp_action = next((a for a in sorted_actions if a.temperature == 22.0), None)
        assert low_temp_action is not None, "Should heat to 22°C during LOW prices"

        # Next should be return to average temp
        actions_after_low = [a for a in sorted_actions if a.timestamp > low_temp_action.timestamp]
        return_to_average = next((a for a in actions_after_low
                                 if a.temperature == 17.0
                                 and "End of LOW price period" in a.reason), None)
        assert return_to_average is not None, "Should return to 17°C when LOW period ends"
        assert return_to_average.timestamp.hour == 16

        # Then should enable ECO mode for PEAK
        eco_action = next((a for a in actions_after_low
                          if a.eco_mode and a.timestamp >= return_to_average.timestamp), None)
        assert eco_action is not None, "Should enable ECO mode for PEAK period"


class TestIntegration:
    """Integration tests with realistic scenarios."""

    @patch('nest_octopus.heating_optimizer.time.sleep')
    def test_full_day_schedule_execution(self, mock_sleep):
        """Test full 24-hour schedule with realistic price data."""
        # Load realistic price data
        daily_prices = load_price_fixture("typical_day_prices.json")
        weekly_prices = load_price_fixture("weekly_prices_sample.json")

        config = Config(
            tariff_code="E-1R-AGILE-FLEX-22-11-25-H",
            thermostat_name="test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj"
        )

        start_time = datetime(2024, 12, 1, 20, 0, tzinfo=timezone.utc)

        actions = calculate_heating_schedule(daily_prices, weekly_prices, config, start_time)

        # Verify schedule characteristics
        assert len(actions) >= 3, "Should have multiple heating changes"

        # Should have low temperature periods (overnight)
        overnight_actions = [a for a in actions if 0 <= a.timestamp.hour < 6]
        if overnight_actions:
            low_temp = [a for a in overnight_actions if a.temperature == 22.0]
            assert len(low_temp) > 0, "Should heat during cheap overnight period"

        # Should have ECO mode during peak (16:00-19:00)
        peak_actions = [a for a in actions if 16 <= a.timestamp.hour < 19]
        if peak_actions:
            eco_actions = [a for a in peak_actions if a.eco_mode]
            assert len(eco_actions) > 0, "Should use ECO during peak hours"

        # Verify all actions have timestamps in the future (from start_time)
        for action in actions:
            assert action.timestamp >= start_time


class TestSignalHandling:
    """Test signal handling for graceful shutdown and config reload."""

    def test_shutdown_signal_handler(self):
        """Test SIGTERM sets shutdown flag."""
        import nest_octopus.heating_optimizer as ho

        # Reset flags
        ho.shutdown_requested = False

        # Call signal handler
        handle_shutdown_signal(signal.SIGTERM, None)

        # Verify flag is set
        assert ho.shutdown_requested is True

    def test_reload_signal_handler(self):
        """Test SIGHUP sets reload flag."""
        import nest_octopus.heating_optimizer as ho

        # Reset flags
        ho.reload_config_requested = False

        # Call signal handler
        handle_reload_signal(signal.SIGHUP, None)

        # Verify flag is set
        assert ho.reload_config_requested is True

    @patch('nest_octopus.heating_optimizer.time.sleep')
    @patch('nest_octopus.heating_optimizer.run_daily_cycle')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py'])
    def test_main_handles_shutdown_signal(self, mock_load_config,
                                         mock_run_cycle, mock_sleep):
        """Test main loop exits gracefully on shutdown signal."""
        import nest_octopus.heating_optimizer as ho
        from nest_octopus.heating_optimizer import main

        # Reset flags
        ho.shutdown_requested = False
        ho.reload_config_requested = False

        # Setup config mock
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        # Simulate shutdown signal after first sleep
        def trigger_shutdown(*args, **kwargs):
            ho.shutdown_requested = True

        mock_sleep.side_effect = trigger_shutdown

        # Run main (should exit after shutdown signal)
        result = main()

        # Verify graceful shutdown
        assert result == 0
        assert ho.shutdown_requested is True

    @patch('nest_octopus.heating_optimizer.time.sleep')
    @patch('nest_octopus.heating_optimizer.run_daily_cycle')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py'])
    def test_main_reloads_config_on_sighup(self, mock_load_config,
                                          mock_run_cycle, mock_sleep):
        """Test main loop reloads configuration on SIGHUP."""
        import nest_octopus.heating_optimizer as ho
        from nest_octopus.heating_optimizer import main

        # Reset flags
        ho.shutdown_requested = False
        ho.reload_config_requested = False

        # Setup config mocks - two different configs
        mock_config1 = Mock(tariff_code="config1")
        mock_config2 = Mock(tariff_code="config2")
        mock_load_config.side_effect = [mock_config1, mock_config2]

        # Simulate reload signal, then shutdown
        call_count = [0]
        def trigger_reload_then_shutdown(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First sleep: trigger reload
                ho.reload_config_requested = True
            elif call_count[0] == 2:
                # Second sleep after reload: trigger shutdown
                ho.shutdown_requested = True

        mock_sleep.side_effect = trigger_reload_then_shutdown

        # Run main
        result = main()

        # Verify config was reloaded
        assert result == 0
        assert mock_load_config.call_count == 2
        assert ho.shutdown_requested is True

    @patch('nest_octopus.heating_optimizer.time.sleep')
    @patch('nest_octopus.heating_optimizer.run_daily_cycle')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py', '--config', '/custom/path/config.ini'])
    def test_main_uses_custom_config_path(self, mock_load_config,
                                         mock_run_cycle, mock_sleep):
        """Test main uses custom config path from --config argument."""
        import nest_octopus.heating_optimizer as ho
        from nest_octopus.heating_optimizer import main

        # Reset flags
        ho.shutdown_requested = False

        # Setup config mock
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        # Trigger shutdown immediately
        def trigger_shutdown(*args, **kwargs):
            ho.shutdown_requested = True

        mock_sleep.side_effect = trigger_shutdown

        # Run main
        result = main()

        # Verify custom config path was used
        assert result == 0
        mock_load_config.assert_called_with('/custom/path/config.ini')

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py', '--dry-run'])
    def test_main_dry_run_mode(self, mock_load_config, mock_octopus_class, capsys):
        """Test main with --dry-run fetches prices and displays schedule."""
        from nest_octopus.heating_optimizer import main

        # Setup config mock
        mock_config = Config(
            tariff_code="E-1R-AGILE-FLEX-22-11-25-H",
            thermostat_name="Test",
            client_id="test-id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj",
            low_price_temp=22.0,
            average_price_temp=17.0,
            peak_start_hour=16,
            peak_end_hour=19
        )
        mock_load_config.return_value = mock_config

        # Setup Octopus client mock
        mock_octopus = Mock()
        mock_octopus_class.return_value = mock_octopus

        daily_prices = load_price_fixture("typical_day_prices.json")
        weekly_prices = load_price_fixture("weekly_prices_sample.json")
        mock_octopus.get_unit_rates.side_effect = [daily_prices, weekly_prices]

        # Run main in dry-run mode
        result = main()

        # Verify successful execution
        assert result == 0

        # Verify prices were fetched
        assert mock_octopus.get_unit_rates.call_count == 2

        # Verify get_unit_rates called with correct parameters
        for call in mock_octopus.get_unit_rates.call_args_list:
            args, kwargs = call
            assert 'tariff_code' in kwargs
            assert 'period_from' in kwargs
            assert 'period_to' in kwargs
            assert 'product_code' not in kwargs

        # Verify output contains key information
        captured = capsys.readouterr()
        assert "DRY RUN MODE" in captured.out
        assert "PRICE ANALYSIS" in captured.out
        assert "PLANNED SCHEDULE" in captured.out
        assert "Dry run complete" in captured.out

        # Verify Octopus client was closed
        mock_octopus.close.assert_called_once()


class TestDryRun:
    """Test dry-run mode functionality."""

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    def test_run_dry_run_success(self, mock_octopus_class, capsys):
        """Test successful dry run execution."""
        # Setup config
        config = Config(
            tariff_code="E-1R-AGILE-FLEX-22-11-25-H",
            thermostat_name="Test",
            client_id="test-id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj"
        )

        # Setup Octopus client mock
        mock_octopus = Mock()
        mock_octopus_class.return_value = mock_octopus

        daily_prices = load_price_fixture("typical_day_prices.json")
        weekly_prices = load_price_fixture("weekly_prices_sample.json")
        mock_octopus.get_unit_rates.side_effect = [daily_prices, weekly_prices]

        # Run dry run
        result = run_dry_run(config)

        # Verify success
        assert result == 0

        # Verify get_unit_rates called with correct parameters
        assert mock_octopus.get_unit_rates.call_count == 2
        for call in mock_octopus.get_unit_rates.call_args_list:
            args, kwargs = call
            assert 'tariff_code' in kwargs
            assert 'period_from' in kwargs
            assert 'period_to' in kwargs
            assert 'product_code' not in kwargs
            assert kwargs['tariff_code'] == config.tariff_code

        # Verify output
        captured = capsys.readouterr()
        assert "DRY RUN MODE" in captured.out
        assert "PRICE ANALYSIS" in captured.out
        assert "Daily Average:" in captured.out
        assert "Weekly Average:" in captured.out
        assert "PLANNED SCHEDULE" in captured.out
        assert "ECO MODE" in captured.out or "HEATING" in captured.out
        assert "Dry run complete" in captured.out

        # Verify client was closed
        mock_octopus.close.assert_called_once()

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    def test_run_dry_run_no_prices(self, mock_octopus_class, capsys):
        """Test dry run when no prices available."""
        config = Config(
            tariff_code="E-1R-AGILE-FLEX-22-11-25-H",
            thermostat_name="Test",
            client_id="test-id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj"
        )

        # Setup Octopus client to return empty prices
        mock_octopus = Mock()
        mock_octopus_class.return_value = mock_octopus
        mock_octopus.get_unit_rates.return_value = []

        # Run dry run
        result = run_dry_run(config)

        # Verify error
        assert result == 1

        # Verify error message
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "No prices available" in captured.out

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    def test_run_dry_run_handles_exception(self, mock_octopus_class, capsys):
        """Test dry run handles exceptions gracefully."""
        config = Config(
            tariff_code="E-1R-AGILE-FLEX-22-11-25-H",
            thermostat_name="Test",
            client_id="test-id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj"
        )

        # Setup Octopus client to raise exception
        mock_octopus = Mock()
        mock_octopus_class.return_value = mock_octopus
        mock_octopus.get_unit_rates.side_effect = Exception("API Error")

        # Run dry run
        result = run_dry_run(config)

        # Verify error
        assert result == 1

        # Verify error message
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

        # Verify client close was attempted
        mock_octopus.close.assert_called_once()

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py', '--dry-run', '--tariff-code', 'E-1R-AGILE-TEST-H'])
    def test_dry_run_with_tariff_code_skips_config(self, mock_load_config, mock_octopus_class, capsys):
        """Test dry-run with --tariff-code doesn't load config file."""
        from nest_octopus.heating_optimizer import main

        # Setup Octopus client mock
        mock_octopus = Mock()
        mock_octopus_class.return_value = mock_octopus

        daily_prices = load_price_fixture("typical_day_prices.json")
        weekly_prices = load_price_fixture("weekly_prices_sample.json")
        mock_octopus.get_unit_rates.side_effect = [daily_prices, weekly_prices]

        # Run main
        result = main()

        # Verify success
        assert result == 0

        # Verify config was NOT loaded
        mock_load_config.assert_not_called()

        # Verify get_unit_rates called with correct parameters
        assert mock_octopus.get_unit_rates.call_count == 2
        for call in mock_octopus.get_unit_rates.call_args_list:
            args, kwargs = call
            assert 'tariff_code' in kwargs
            assert 'period_from' in kwargs
            assert 'period_to' in kwargs
            assert 'product_code' not in kwargs
            assert kwargs['tariff_code'] == 'E-1R-AGILE-TEST-H'

        # Verify output contains tariff code
        captured = capsys.readouterr()
        assert "E-1R-AGILE-TEST-H" in captured.out
        assert "DRY RUN MODE" in captured.out

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py', '--dry-run', '--tariff-code', 'E-1R-OVERRIDE-H'])
    def test_tariff_code_overrides_config(self, mock_load_config, mock_octopus_class, capsys):
        """Test --tariff-code overrides config file setting."""
        from nest_octopus.heating_optimizer import main

        # Setup config with different tariff code
        mock_config = Config(
            tariff_code="E-1R-AGILE-ORIGINAL-H",
            thermostat_name="Test",
            client_id="test-id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj"
        )
        mock_load_config.return_value = mock_config

        # Setup Octopus client mock
        mock_octopus = Mock()
        mock_octopus_class.return_value = mock_octopus

        daily_prices = load_price_fixture("typical_day_prices.json")
        weekly_prices = load_price_fixture("weekly_prices_sample.json")
        mock_octopus.get_unit_rates.side_effect = [daily_prices, weekly_prices]

        # Run main
        result = main()

        # Verify success
        assert result == 0

        # Verify get_unit_rates called with override tariff code
        assert mock_octopus.get_unit_rates.call_count == 2
        for call in mock_octopus.get_unit_rates.call_args_list:
            args, kwargs = call
            assert 'tariff_code' in kwargs
            assert 'period_from' in kwargs
            assert 'period_to' in kwargs
            assert 'product_code' not in kwargs
            assert kwargs['tariff_code'] == 'E-1R-OVERRIDE-H'

        # Verify config was loaded (no --config, no --tariff-code doesn't skip it)
        # Actually in this case it should be loaded because we have --dry-run but not ONLY --tariff-code
        # Wait, let me re-read the logic...
        # If dry-run AND tariff-code, skip config
        # So config should NOT be loaded
        mock_load_config.assert_not_called()

        # Verify override tariff code was used
        captured = capsys.readouterr()
        assert "E-1R-OVERRIDE-H" in captured.out
        assert "DRY RUN MODE" in captured.out

    @patch('nest_octopus.heating_optimizer.time.sleep')
    @patch('nest_octopus.heating_optimizer.run_daily_cycle')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py', '--config', '/test/config.ini', '--tariff-code', 'E-1R-OVERRIDE-H'])
    def test_tariff_code_overrides_in_daemon_mode(self, mock_load_config, mock_run_cycle, mock_sleep):
        """Test --tariff-code overrides config in normal daemon mode."""
        import nest_octopus.heating_optimizer as ho
        from nest_octopus.heating_optimizer import main

        # Reset flags
        ho.shutdown_requested = False

        # Setup config with different tariff code
        mock_config = Mock(tariff_code="E-1R-AGILE-ORIGINAL-H")
        mock_load_config.return_value = mock_config

        # Trigger shutdown immediately
        def trigger_shutdown(*args, **kwargs):
            ho.shutdown_requested = True

        mock_sleep.side_effect = trigger_shutdown

        # Run main
        result = main()

        # Verify success
        assert result == 0

        # Verify config was loaded
        mock_load_config.assert_called_with('/test/config.ini')

        # Verify tariff code was overridden
        assert mock_config.tariff_code == "E-1R-OVERRIDE-H"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


