# SPDX-License-Identifier: MPL-2.0
"""
Comprehensive unit tests for the heating optimization daemon.

Tests configuration loading, price analysis, scheduling algorithm,
and thermostat control with mocked REST endpoints and time functions.
"""

import asyncio
import json
import os
import signal
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List
from unittest.mock import Mock, patch, MagicMock

import pytest

from nest_octopus.heating_optimizer import (
    Config,
    ConfigurationError,
    HeatingAction,
    TemperatureTier,
    calculate_heating_schedule,
    calculate_price_statistics,
    determine_target_temperature,
    execute_heating_action,
    find_cheapest_windows,
    find_default_config,
    load_config,
    parse_temperature_tier,
    parse_quiet_window,
    parse_tg_active_period,
    run_daily_cycle,
    run_dry_run,
)
from nest_octopus.octopus import PricePoint


# Fixture directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "heating"


def create_price_point(valid_from_str: str, valid_to_str: str, price_inc_vat: float) -> PricePoint:
    """Helper to create a PricePoint for testing."""
    return PricePoint({
        'valid_from': valid_from_str,
        'valid_to': valid_to_str,
        'value_exc_vat': price_inc_vat / 1.05,
        'value_inc_vat': price_inc_vat
    })


def load_price_fixture(filename: str) -> List[PricePoint]:
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

    def test_load_config_success(self, tmp_path: Any) -> None:
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
low_price_temp = 20.0
average_price_temp = 17.0
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
        # Check default temperature tiers
        assert len(config.temperature_tiers) >= 1
        assert config.default_temp == 17.0

    def test_load_config_missing_file(self) -> None:
        """Test error when config file doesn't exist."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config("nonexistent.ini")

    def test_load_config_no_credentials_dir(self, tmp_path: Any) -> None:
        """Test error when CREDENTIALS_DIRECTORY not set."""
        config_file = tmp_path / "config.ini"
        config_file.write_text("[octopus]\ntariff_code = TEST\n")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="CREDENTIALS_DIRECTORY"):
                load_config(str(config_file))

    def test_load_config_missing_credentials(self, tmp_path: Any) -> None:
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

    def test_load_config_default_heating_values(self, tmp_path: Any) -> None:
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
        assert len(config.temperature_tiers) >= 1
        assert config.default_temp == 17.0

    def test_find_default_config_not_found(self) -> None:
        """Test find_default_config returns None when no files exist."""
        with patch('os.path.exists', return_value=False):
            result = find_default_config()
            assert result is None

    def test_find_default_config_finds_etc(self) -> None:
        """Test find_default_config finds /etc config first."""
        def exists_mock(path: Any) -> bool:
            return bool(path == "/etc/nest-octopus/nest-octopus.conf")

        with patch('os.path.exists', side_effect=exists_mock):
            result = find_default_config()
            assert result == "/etc/nest-octopus/nest-octopus.conf"

    def test_find_default_config_finds_run(self) -> None:
        """Test find_default_config finds /run config if /etc missing."""
        def exists_mock(path: Any) -> bool:
            return bool(path == "/run/nest-octopus/nest-octopus.conf")

        with patch('os.path.exists', side_effect=exists_mock):
            result = find_default_config()
            assert result == "/run/nest-octopus/nest-octopus.conf"

    def test_find_default_config_finds_usr_lib(self) -> None:
        """Test find_default_config finds /usr/lib config if others missing."""
        def exists_mock(path: Any) -> bool:
            return bool(path == "/usr/lib/nest-octopus/nest-octopus.conf")

        with patch('os.path.exists', side_effect=exists_mock):
            result = find_default_config()
            assert result == "/usr/lib/nest-octopus/nest-octopus.conf"

    def test_load_config_with_none_path_searches(self, tmp_path: Any) -> None:
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

    def test_load_config_with_none_path_fails_if_not_found(self) -> None:
        """Test load_config raises error when no default config found."""
        with patch('nest_octopus.heating_optimizer.find_default_config', return_value=None):
            with pytest.raises(ConfigurationError, match="No configuration file found"):
                load_config(None)


class TestHeatingSchedule:
    """Test heating schedule calculation."""

    def test_calculate_heating_schedule_typical_day(self) -> None:
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
            temperature_tiers=[TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)],
            default_temp=17.0
        )

        start_time = datetime(2024, 12, 1, 20, 0)
        actions = calculate_heating_schedule(daily_prices, weekly_prices, config, start_time)

        # Should have multiple actions
        assert len(actions) > 0

        # Actions should be sorted by time
        for i in range(len(actions) - 1):
            assert actions[i].timestamp <= actions[i + 1].timestamp

        # Should have ECO mode during HIGH prices
        eco_actions = [a for a in actions if a.eco_mode]
        assert len(eco_actions) > 0, "Should use ECO during high price periods"

        # Should have low temperature setpoint during low price periods
        low_temp_actions = [a for a in actions if a.temperature == 20.0]
        assert len(low_temp_actions) > 0, "Should heat to 20°C during low prices"

    def test_heating_action_representation(self) -> None:
        """Test HeatingAction string representation."""
        eco_action = HeatingAction(
            timestamp=datetime(2024, 12, 2, 17, 0),
            temperature=None,
            eco_mode=True,
            reason="Peak price"
        )

        temp_action = HeatingAction(
            timestamp=datetime(2024, 12, 2, 2, 0),
            temperature=20.0,
            eco_mode=False,
            reason="Low price"
        )

        assert "ECO" in repr(eco_action)
        assert "20" in repr(temp_action)


class TestThermostatControl:
    """Test thermostat control functions."""

    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    def test_execute_heating_action_eco_mode(self, mock_client_class: Any) -> None:
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

        execute_heating_action(action, mock_client)

        # Should call set_eco_mode with MANUAL_ECO
        mock_client.set_eco_mode.assert_called_once_with(EcoMode.MANUAL_ECO)

    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    def test_execute_heating_action_temperature(self, mock_client_class: Any) -> None:
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
                'sdm.devices.traits.ThermostatEco': {'mode': 'MANUAL_ECO'},
                'sdm.devices.traits.ThermostatHvac': {'status': 'OFF'},
                'sdm.devices.traits.ThermostatTemperatureSetpoint': {'heatCelsius': 20.0}
            }
        }
        mock_status = ThermostatStatus(device_data)
        mock_client.get_device.return_value = mock_status

        action = HeatingAction(
            timestamp=datetime(2024, 12, 2, 2, 0),
            temperature=20.0,
            eco_mode=False,
            reason="Low price"
        )

        execute_heating_action(action, mock_client)

        # Should disable ECO, set HEAT mode, then set temperature
        assert mock_client.set_eco_mode.call_count >= 1
        mock_client.set_heat.assert_called_once_with(20.0)


class TestDailyCycle:
    """Test the daily optimization cycle."""

    @pytest.mark.asyncio
    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    async def test_run_daily_cycle_complete(self, mock_octopus_class: Any,
                                     mock_nest_class: Any) -> None:
        """Test complete daily cycle execution."""
        from nest_octopus.nest_thermostat import ThermostatStatus

        # Mock Octopus client
        mock_octopus = Mock()
        mock_octopus.__enter__ = Mock(return_value=mock_octopus)
        mock_octopus.__exit__ = Mock(return_value=False)
        mock_octopus_class.return_value = mock_octopus

        daily_prices = load_price_fixture("typical_day_prices.json")
        weekly_prices = load_price_fixture("weekly_prices_sample.json")

        mock_octopus.get_unit_rates.side_effect = [daily_prices, weekly_prices]

        # Mock Nest client
        mock_nest = Mock()
        mock_nest.device_id = "enterprises/proj/devices/thermostat-123"
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

        # Run cycle with nest client
        handles = await run_daily_cycle(config, mock_nest)

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

        # Verify handles are returned (may be empty if all actions are in the past)
        assert isinstance(handles, list)

        # Cancel all scheduled actions
        for handle in handles:
            handle.cancel()

        # Nest client should not be closed by run_daily_cycle
        assert not mock_nest.close.called

    @pytest.mark.asyncio
    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    async def test_run_daily_cycle_no_prices(self, mock_octopus_class: Any,
                                       mock_nest_class: Any) -> None:
        """Test handling when no prices are available."""
        mock_octopus = Mock()
        mock_octopus.__enter__ = Mock(return_value=mock_octopus)
        mock_octopus.__exit__ = Mock(return_value=False)
        mock_octopus_class.return_value = mock_octopus
        mock_octopus.get_unit_rates.return_value = []  # No prices

        mock_nest = Mock()
        mock_nest.device_id = "test-device-id"
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

        handles = await run_daily_cycle(config, mock_nest)

        # Should return empty handles
        assert handles == []

        # Should not attempt to control thermostat
        assert not mock_nest.set_mode.called
        assert not mock_nest.set_heat.called

        # Nest client should not be closed by run_daily_cycle
        assert not mock_nest.close.called


class TestSchedulingLogic:
    """Test specific scheduling scenarios."""

    def test_overnight_low_prices_heat_to_22(self) -> None:
        """Test that overnight low prices result in heating to 22°C."""
        # Create overnight low price scenario
        # Need more price points to establish proper averages
        prices = [
            create_price_point('2024-12-02T00:00:00Z', '2024-12-02T00:30:00Z', 15.0),  # AVERAGE
            create_price_point('2024-12-02T01:00:00Z', '2024-12-02T01:30:00Z', 18.0),  # AVERAGE
            create_price_point('2024-12-02T02:00:00Z', '2024-12-02T02:30:00Z', 8.0),   # LOW
            create_price_point('2024-12-02T02:30:00Z', '2024-12-02T03:00:00Z', 7.0),   # LOW
            create_price_point('2024-12-02T03:00:00Z', '2024-12-02T03:30:00Z', 16.0),  # AVERAGE
        ]

        weekly_prices = [
            create_price_point('2024-11-25T02:00:00Z', '2024-11-25T02:30:00Z', 15.0)
        ]

        config = Config(
            tariff_code="TEST",
            thermostat_name="test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj",
            temperature_tiers=[TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)],
            default_temp=17.0
        )

        actions = calculate_heating_schedule(
            prices, weekly_prices, config, datetime(2024, 12, 1, 20, 0, tzinfo=timezone.utc)
        )

        # Should have action to heat to 20°C
        low_temp_actions = [a for a in actions if a.temperature == 20.0]
        assert len(low_temp_actions) > 0

    def test_high_prices_use_eco(self) -> None:
        """Test that HIGH prices trigger ECO mode."""
        # Create HIGH price scenario (prices well above average)
        prices = [
            create_price_point('2024-12-02T10:00:00Z', '2024-12-02T10:30:00Z', 15.0),  # AVERAGE
            create_price_point('2024-12-02T16:00:00Z', '2024-12-02T16:30:00Z', 45.0),  # HIGH
            create_price_point('2024-12-02T17:00:00Z', '2024-12-02T17:30:00Z', 50.0),  # HIGH
            create_price_point('2024-12-02T18:00:00Z', '2024-12-02T18:30:00Z', 48.0),  # HIGH
            create_price_point('2024-12-02T19:00:00Z', '2024-12-02T19:30:00Z', 20.0),  # AVERAGE
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
            project_id="proj"
        )

        actions = calculate_heating_schedule(
            prices, weekly_prices, config, datetime(2024, 12, 1, 20, 0, tzinfo=timezone.utc)
        )

        # Should have ECO mode actions during HIGH price periods
        eco_actions = [a for a in actions if a.eco_mode]
        assert len(eco_actions) > 0, "Should enable ECO mode during HIGH prices"

    def test_low_price_window_ends_returns_to_average_temp(self) -> None:
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
            temperature_tiers=[TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)],
            default_temp=17.0
        )

        actions = calculate_heating_schedule(
            prices, weekly_prices, config, datetime(2024, 12, 1, 20, 0, tzinfo=timezone.utc)
        )

        # Find the action that sets temperature to 20°C (entering tier during low prices)
        low_temp_action = next((a for a in actions if a.temperature == 20.0), None)
        assert low_temp_action is not None, "Should have action to heat to 20°C during low prices"

        # Find the action that returns to 17°C (exiting tier when prices rise)
        return_to_default = [a for a in actions
                            if a.temperature == 17.0
                            and a.timestamp > low_temp_action.timestamp
                            and "Temperature tier" in a.reason]
        assert len(return_to_default) > 0, "Should return to 17°C when prices rise above tier threshold"

        # Verify the return happens at the expected time (03:00 when prices return to average)
        assert return_to_default[0].timestamp.hour == 3
        assert return_to_default[0].timestamp.minute == 0

    def test_low_price_to_high_returns_to_average_then_eco(self) -> None:
        """Test that when tier period transitions to ECO, it goes directly to ECO mode."""
        # Create scenario: tier prices 14:00-16:00, then ECO at 16:00
        prices = [
            create_price_point('2024-12-02T14:00:00Z', '2024-12-02T14:30:00Z', 8.0),   # Tier
            create_price_point('2024-12-02T14:30:00Z', '2024-12-02T15:00:00Z', 7.0),   # Tier
            create_price_point('2024-12-02T15:00:00Z', '2024-12-02T15:30:00Z', 9.0),   # Tier
            create_price_point('2024-12-02T15:30:00Z', '2024-12-02T16:00:00Z', 8.5),   # Tier
            create_price_point('2024-12-02T16:00:00Z', '2024-12-02T16:30:00Z', 45.0),  # ECO
            create_price_point('2024-12-02T16:30:00Z', '2024-12-02T17:00:00Z', 50.0),  # ECO
            create_price_point('2024-12-02T17:00:00Z', '2024-12-02T17:30:00Z', 48.0),  # ECO
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
            temperature_tiers=[TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)],
            default_temp=17.0
        )

        actions = calculate_heating_schedule(
            prices, weekly_prices, config, datetime(2024, 12, 1, 20, 0, tzinfo=timezone.utc)
        )

        # Find the sequence of actions
        sorted_actions = sorted(actions, key=lambda a: a.timestamp)

        # Should have: 20°C (tier), then ECO (high prices)
        tier_temp_action = next((a for a in sorted_actions if a.temperature == 20.0), None)
        assert tier_temp_action is not None, "Should heat to 20°C during tier prices"

        # Next should enable ECO mode for high prices (directly, no intermediate default temp)
        actions_after_tier = [a for a in sorted_actions if a.timestamp > tier_temp_action.timestamp]
        eco_action = next((a for a in actions_after_tier if a.eco_mode), None)
        assert eco_action is not None, "Should enable ECO mode for high price period"
        assert eco_action.timestamp.hour == 16

    def test_disabling_eco_mode_does_not_set_temperature(self) -> None:
        """Test that when exiting HIGH period (ECO mode), temperature is not set when entering AVERAGE."""
        # Create scenario: AVERAGE → HIGH → AVERAGE → LOW
        # When transitioning HIGH→AVERAGE, ECO should be disabled WITHOUT setting temperature
        prices = [
            create_price_point('2024-12-08T08:00:00Z', '2024-12-08T08:30:00Z', 18.0),  # AVERAGE
            create_price_point('2024-12-08T08:30:00Z', '2024-12-08T09:00:00Z', 50.0),  # HIGH
            create_price_point('2024-12-08T09:00:00Z', '2024-12-08T09:30:00Z', 45.0),  # HIGH
            create_price_point('2024-12-08T09:30:00Z', '2024-12-08T10:00:00Z', 16.0),  # AVERAGE
            create_price_point('2024-12-08T10:00:00Z', '2024-12-08T10:30:00Z', 17.0),  # AVERAGE
            create_price_point('2024-12-08T10:30:00Z', '2024-12-08T11:00:00Z', 8.0),   # LOW
        ]

        weekly_prices = [
            create_price_point('2024-12-01T10:00:00Z', '2024-12-01T10:30:00Z', 20.0)
        ]

        config = Config(
            tariff_code="TEST",
            thermostat_name="test",
            client_id="id",
            client_secret="secret",
            refresh_token="token",
            project_id="proj",
            temperature_tiers=[TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)],
            default_temp=17.0
        )

        actions = calculate_heating_schedule(
            prices, weekly_prices, config, datetime(2024, 12, 7, 20, 0, tzinfo=timezone.utc)
        )

        sorted_actions = sorted(actions, key=lambda a: a.timestamp)

        # Action 1: Should set temperature to 17°C at start (AVERAGE period)
        action_08_00 = next((a for a in sorted_actions if a.timestamp.hour == 8 and a.timestamp.minute == 0), None)
        assert action_08_00 is not None, "Should have action at 08:00"
        assert action_08_00.temperature == 17.0, "Should set to 17°C during initial AVERAGE period"
        assert not action_08_00.eco_mode, "Should not enable ECO mode"

        # Action 2: Should enable ECO mode at 08:30 (HIGH period)
        action_08_30 = next((a for a in sorted_actions if a.timestamp.hour == 8 and a.timestamp.minute == 30), None)
        assert action_08_30 is not None, "Should have action at 08:30"
        assert action_08_30.eco_mode, "Should enable ECO mode during HIGH prices"
        assert action_08_30.temperature is None, "ECO mode action should not set temperature"

        # Action 3: Should ONLY disable ECO mode at 09:30, WITHOUT setting temperature
        action_09_30 = next((a for a in sorted_actions if a.timestamp.hour == 9 and a.timestamp.minute == 30), None)
        assert action_09_30 is not None, "Should have action at 09:30"
        assert not action_09_30.eco_mode, "Should disable ECO mode"
        assert action_09_30.temperature is None, "Should NOT set temperature when disabling ECO mode"
        assert "End of HIGH price period" in action_09_30.reason, "Reason should indicate end of HIGH period"

        # Action 4: Should set temperature to 20°C at 10:30 (LOW period)
        action_10_30 = next((a for a in sorted_actions if a.timestamp.hour == 10 and a.timestamp.minute == 30), None)
        assert action_10_30 is not None, "Should have action at 10:30"
        assert action_10_30.temperature == 20.0, "Should set to 20°C during LOW period"
        assert not action_10_30.eco_mode, "Should not enable ECO mode during LOW prices"

        # Verify NO action sets temperature during AVERAGE period after ECO disable
        average_after_eco = [a for a in sorted_actions
                            if a.timestamp.hour == 9 and a.timestamp.minute == 30
                            or a.timestamp.hour == 10 and a.timestamp.minute == 0]
        for action in average_after_eco:
            if action.timestamp.hour == 9:  # The ECO disable action
                assert action.temperature is None, "ECO disable should not set temperature"
            # No action at 10:00 since we're staying in AVERAGE after POST_ECO state

        # Total actions should be: 08:00 (17°C), 08:30 (ECO), 09:30 (disable ECO), 10:30 (20°C)
        assert len(sorted_actions) == 4, f"Should have exactly 4 actions, got {len(sorted_actions)}"


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_day_schedule_execution(self) -> None:
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
            low_temp = [a for a in overnight_actions if a.temperature == 20.0]
            assert len(low_temp) > 0, "Should heat during cheap overnight period"

        # Should have ECO mode during HIGH prices
        eco_actions = [a for a in actions if a.eco_mode]
        assert len(eco_actions) > 0, "Should use ECO during high price periods"

        # Verify all actions have timestamps in the future (from start_time)
        for action in actions:
            assert action.timestamp >= start_time


class TestSignalHandling:
    """Test signal handling for graceful shutdown and config reload."""

    # Note: Signal handler tests removed as signals are now handled via asyncio.Event
    # The signal handlers are internal to wait_for_signal() and tested via integration tests

    @pytest.mark.asyncio
    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    @patch('nest_octopus.heating_optimizer.run_daily_cycle')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py'])
    async def test_main_handles_shutdown_signal(self, mock_load_config: Any,
                                         mock_run_cycle: Any, mock_nest_class: Any) -> None:
        """Test main loop exits gracefully on shutdown signal."""
        from nest_octopus.heating_optimizer import async_main

        # Setup config mock
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        # Setup nest client mock
        mock_nest = Mock()
        mock_nest.device_id = "test-device-id"
        mock_nest_class.return_value = mock_nest

        # Make run_daily_cycle async and return empty handles
        async def mock_daily_cycle(*args: Any, **kwargs: Any) -> List[Any]:
            return []
        mock_run_cycle.side_effect = mock_daily_cycle

        # Simulate shutdown signal after first wait
        call_count = [0]
        async def mock_wait(tasks: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                # First wait: return timeout (empty done set)
                return (set(), set(tasks))
            # Subsequent waits should not occur as shutdown will happen
            return (set(), set(tasks))

        with patch('nest_octopus.heating_optimizer.asyncio.wait', side_effect=mock_wait):
            # Patch the signal handler setup to capture events and trigger shutdown
            def mock_signal_setup(shutdown_event: asyncio.Event, reload_event: asyncio.Event) -> None:
                # Set shutdown immediately to exit loop after first cycle
                shutdown_event.set()

            with patch('nest_octopus.heating_optimizer.setup_signal_handlers', side_effect=mock_signal_setup):
                # Run main (should exit after shutdown signal)
                result = await async_main()

        # Verify graceful shutdown
        assert result == 0

    @pytest.mark.asyncio
    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    @patch('nest_octopus.heating_optimizer.run_daily_cycle')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py'])
    async def test_main_reloads_config_on_sighup(self, mock_load_config: Any,
                                          mock_run_cycle: Any, mock_nest_class: Any) -> None:
        """Test main loop reloads configuration on SIGHUP."""
        from nest_octopus.heating_optimizer import async_main

        # Setup config mocks - two different configs
        mock_config1 = Mock(tariff_code="config1")
        mock_config2 = Mock(tariff_code="config2")
        mock_load_config.side_effect = [mock_config1, mock_config2]

        # Setup nest client mock - need to track multiple instances
        mock_nest1 = Mock()
        mock_nest1.device_id = "test-device-id-1"
        mock_nest2 = Mock()
        mock_nest2.device_id = "test-device-id-2"
        mock_nest_class.side_effect = [mock_nest1, mock_nest2]

        # Make run_daily_cycle async and return empty handles
        async def mock_daily_cycle(*args: Any, **kwargs: Any) -> List[Any]:
            return []
        mock_run_cycle.side_effect = mock_daily_cycle

        # Track how many times we've been called
        wait_count = [0]
        reload_event_ref: List[Any] = [None]
        shutdown_event_ref: List[Any] = [None]

        async def mock_wait(tasks: Any, **kwargs: Any) -> Any:
            wait_count[0] += 1
            if wait_count[0] == 1:
                # First wait: trigger reload
                if reload_event_ref[0]:
                    reload_event_ref[0].set()
                # Return empty done, all tasks pending (timeout)
                return (set(), set(tasks))
            elif wait_count[0] == 2:
                # After reload, trigger shutdown
                if shutdown_event_ref[0]:
                    shutdown_event_ref[0].set()
                # Return empty done, all tasks pending (timeout)
                return (set(), set(tasks))
            # Default: return empty done, all tasks pending
            return (set(), set(tasks))

        with patch('nest_octopus.heating_optimizer.asyncio.wait', side_effect=mock_wait):
            # Capture events for manipulation
            def mock_signal_setup(shutdown_event: asyncio.Event, reload_event: asyncio.Event) -> None:
                reload_event_ref[0] = reload_event
                shutdown_event_ref[0] = shutdown_event

            with patch('nest_octopus.heating_optimizer.setup_signal_handlers', side_effect=mock_signal_setup):
                # Run main
                result = await async_main()

        # Verify config was reloaded
        assert result == 0
        assert mock_load_config.call_count == 2
        # Verify first nest client was closed and second was created
        assert mock_nest_class.call_count == 2
        mock_nest1.close.assert_called_once()
        mock_nest2.close.assert_called_once()

    @pytest.mark.asyncio
    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    @patch('nest_octopus.heating_optimizer.run_daily_cycle')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py', '--config', '/custom/path/config.ini'])
    async def test_main_uses_custom_config_path(self, mock_load_config: Any,
                                         mock_run_cycle: Any, mock_nest_class: Any) -> None:
        """Test main uses custom config path from --config argument."""
        from nest_octopus.heating_optimizer import async_main

        # Setup config mock
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        # Setup nest client mock
        mock_nest = Mock()
        mock_nest.device_id = "test-device-id"
        mock_nest_class.return_value = mock_nest

        # Make run_daily_cycle async and return empty handles
        async def mock_daily_cycle(*args: Any, **kwargs: Any) -> List[Any]:
            return []
        mock_run_cycle.side_effect = mock_daily_cycle

        # Trigger shutdown immediately
        def mock_signal_setup(shutdown_event: asyncio.Event, reload_event: asyncio.Event) -> None:
            shutdown_event.set()

        with patch('nest_octopus.heating_optimizer.setup_signal_handlers', side_effect=mock_signal_setup):
            # Run main
            result = await async_main()

        # Verify custom config path was used
        assert result == 0
        mock_load_config.assert_called_with('/custom/path/config.ini')

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py', '--dry-run'])
    def test_main_dry_run_mode(self, mock_load_config: Any, mock_octopus_class: Any, capsys: Any) -> None:
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
            temperature_tiers=[TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)],
            default_temp=17.0
        )
        mock_load_config.return_value = mock_config

        # Setup Octopus client mock with context manager support
        mock_octopus = Mock()
        mock_octopus.__enter__ = Mock(return_value=mock_octopus)
        mock_octopus.__exit__ = Mock(return_value=False)
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


class TestDryRun:
    """Test dry-run mode functionality."""

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    def test_run_dry_run_success(self, mock_octopus_class: Any, capsys: Any) -> None:
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
        mock_octopus.__enter__ = Mock(return_value=mock_octopus)
        mock_octopus.__exit__ = Mock(return_value=False)
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

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    def test_run_dry_run_no_prices(self, mock_octopus_class: Any, capsys: Any) -> None:
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
        mock_octopus.__enter__ = Mock(return_value=mock_octopus)
        mock_octopus.__exit__ = Mock(return_value=False)
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
    def test_run_dry_run_handles_exception(self, mock_octopus_class: Any, capsys: Any) -> None:
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
        mock_octopus.__enter__ = Mock(return_value=mock_octopus)
        mock_octopus.__exit__ = Mock(return_value=False)
        mock_octopus_class.return_value = mock_octopus
        mock_octopus.get_unit_rates.side_effect = Exception("API Error")

        # Run dry run
        result = run_dry_run(config)

        # Verify error
        assert result == 1

        # Verify error message
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    @patch('nest_octopus.heating_optimizer.OctopusEnergyClient')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py', '--dry-run', '--tariff-code', 'E-1R-AGILE-TEST-H'])
    def test_dry_run_with_tariff_code_skips_config(self, mock_load_config: Any, mock_octopus_class: Any, capsys: Any) -> None:
        """Test dry-run with --tariff-code doesn't load config file."""
        from nest_octopus.heating_optimizer import main

        # Setup Octopus client mock
        mock_octopus = Mock()
        mock_octopus.__enter__ = Mock(return_value=mock_octopus)
        mock_octopus.__exit__ = Mock(return_value=False)
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
    def test_tariff_code_overrides_config(self, mock_load_config: Any, mock_octopus_class: Any, capsys: Any) -> None:
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
        mock_octopus.__enter__ = Mock(return_value=mock_octopus)
        mock_octopus.__exit__ = Mock(return_value=False)
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

    @pytest.mark.asyncio
    @patch('nest_octopus.heating_optimizer.NestThermostatClient')
    @patch('nest_octopus.heating_optimizer.run_daily_cycle')
    @patch('nest_octopus.heating_optimizer.load_config')
    @patch('sys.argv', ['heating_optimizer.py', '--config', '/test/config.ini', '--tariff-code', 'E-1R-OVERRIDE-H'])
    async def test_tariff_code_overrides_in_daemon_mode(self, mock_load_config: Any, mock_run_cycle: Any, mock_nest_class: Any) -> None:
        """Test --tariff-code overrides config in normal daemon mode."""
        from nest_octopus.heating_optimizer import async_main

        # Setup config with different tariff code
        mock_config = Mock(tariff_code="E-1R-AGILE-ORIGINAL-H")
        mock_load_config.return_value = mock_config

        # Setup nest client mock
        mock_nest = Mock()
        mock_nest.device_id = "test-device-id"
        mock_nest_class.return_value = mock_nest

        # Make run_daily_cycle async and return empty handles
        async def mock_daily_cycle(*args: Any, **kwargs: Any) -> List[Any]:
            return []
        mock_run_cycle.side_effect = mock_daily_cycle

        # Trigger shutdown immediately
        def mock_signal_setup(shutdown_event: asyncio.Event, reload_event: asyncio.Event) -> None:
            shutdown_event.set()

        with patch('nest_octopus.heating_optimizer.setup_signal_handlers', side_effect=mock_signal_setup):
            # Run main
            result = await async_main()

        # Verify success
        assert result == 0

        # Verify config was loaded
        mock_load_config.assert_called_with('/test/config.ini')

        # Verify tariff code was overridden
        assert mock_config.tariff_code == "E-1R-OVERRIDE-H"


class TestConfigValidation:
    """Test configuration validation edge cases."""

    def test_load_config_missing_tariff_and_credentials(self, tmp_path: Any) -> None:
        """Test error when neither tariff_code nor api credentials are provided."""
        config_file = tmp_path / "config.ini"
        config_file.write_text("""
[octopus]
# No tariff_code, no api_key, no account_number

[nest]
thermostat_name = Living Room
client_id = test-client-id.apps.googleusercontent.com
project_id = test-project-123

[heating]
low_price_temp = 20.0
""")

        creds_dir = tmp_path / "credentials"
        creds_dir.mkdir()
        (creds_dir / "client_secret").write_text("test-secret")
        (creds_dir / "refresh_token").write_text("test-refresh")

        with patch.dict(os.environ, {'CREDENTIALS_DIRECTORY': str(creds_dir)}):
            with pytest.raises(ConfigurationError, match="Either tariff_code or both api_key and account_number must be configured"):
                load_config(str(config_file))

    def test_load_config_only_api_key_no_account(self, tmp_path: Any) -> None:
        """Test error when only api_key is provided without account_number."""
        config_file = tmp_path / "config.ini"
        config_file.write_text("""
[octopus]
account_number = A-12345678

[nest]
thermostat_name = Living Room
client_id = test-client-id.apps.googleusercontent.com
project_id = test-project-123
""")

        creds_dir = tmp_path / "credentials"
        creds_dir.mkdir()
        (creds_dir / "client_secret").write_text("test-secret")
        (creds_dir / "refresh_token").write_text("test-refresh")
        # No octopus_api_key file

        with patch.dict(os.environ, {'CREDENTIALS_DIRECTORY': str(creds_dir)}):
            with pytest.raises(ConfigurationError, match="Either tariff_code or both api_key and account_number must be configured"):
                load_config(str(config_file))

    def test_load_config_invalid_heating_values(self, tmp_path: Any) -> None:
        """Test handling of invalid heating temperature values."""
        config_file = tmp_path / "config.ini"
        config_file.write_text("""
[octopus]
tariff_code = E-1R-AGILE-FLEX-22-11-25-H

[nest]
thermostat_name = Living Room
client_id = test-client-id.apps.googleusercontent.com
project_id = test-project-123

[heating]
tier1 = 20@75%
default_temp = invalid
""")

        creds_dir = tmp_path / "credentials"
        creds_dir.mkdir()
        (creds_dir / "client_secret").write_text("test-secret")
        (creds_dir / "refresh_token").write_text("test-refresh")

        with patch.dict(os.environ, {'CREDENTIALS_DIRECTORY': str(creds_dir)}):
            with pytest.raises(ValueError):
                load_config(str(config_file))


class TestQuietWindow:
    """Test quiet window functionality."""

    def test_parse_quiet_window_normal(self) -> None:
        """Test parsing a normal time range (not crossing midnight)."""
        result = parse_quiet_window("09:00-17:00")
        assert result == (9, 0, 17, 0)

    def test_parse_quiet_window_midnight_crossing(self) -> None:
        """Test parsing a time range that crosses midnight."""
        result = parse_quiet_window("23:00-07:00")
        assert result == (23, 0, 7, 0)

    def test_parse_quiet_window_with_minutes(self) -> None:
        """Test parsing a time range with minutes."""
        result = parse_quiet_window("22:30-06:45")
        assert result == (22, 30, 6, 45)

    def test_parse_quiet_window_spaces(self) -> None:
        """Test parsing with extra spaces."""
        result = parse_quiet_window("  09:00 - 17:00  ")
        assert result == (9, 0, 17, 0)

    def test_parse_quiet_window_invalid_format(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Expected 'hh:mm-hh:mm'"):
            parse_quiet_window("invalid")

    def test_parse_quiet_window_invalid_hour(self) -> None:
        """Test that invalid hour raises ValueError."""
        with pytest.raises(ValueError, match="Start hour must be 0-23"):
            parse_quiet_window("25:00-07:00")

    def test_parse_quiet_window_invalid_minute(self) -> None:
        """Test that invalid minute raises ValueError."""
        with pytest.raises(ValueError, match="Start minute must be 0-59"):
            parse_quiet_window("09:60-17:00")

    def test_parse_quiet_window_missing_colon(self) -> None:
        """Test that missing colon raises ValueError."""
        with pytest.raises(ValueError):
            parse_quiet_window("0900-1700")

    def test_calculate_heating_schedule_with_quiet_window_normal(self) -> None:
        """Test that actions during a normal quiet window are filtered out."""
        # Create prices that would generate actions during 09:00-17:00
        prices = []
        base_time = datetime(2024, 1, 1, 20, 0, tzinfo=timezone.utc)

        # Create alternating low/high prices to generate many actions
        for i in range(48):  # 24 hours
            hour_offset = i // 2
            minute = (i % 2) * 30
            valid_from = base_time + timedelta(hours=hour_offset, minutes=minute)
            valid_to = valid_from + timedelta(minutes=30)

            # Alternate between low (3p) and high (16p) prices
            price = 3.0 if (i // 4) % 2 == 0 else 16.0

            prices.append(create_price_point(
                valid_from.isoformat(),
                valid_to.isoformat(),
                price
            ))

        weekly_prices = [create_price_point(
            '2023-12-25T00:00:00Z',
            '2023-12-25T00:30:00Z',
            10.0
        )] * 336

        config = Config(
            thermostat_name='Test',
            client_id='test',
            client_secret='test',
            refresh_token='test',
            project_id='test',
            tariff_code='AGILE-TEST',
            quiet_window=(9, 0, 17, 0)  # 09:00-17:00
        )

        actions = calculate_heating_schedule(prices, weekly_prices, config, base_time)

        # Check that no temperature-setting actions fall within 09:00-17:00 local time
        # But ECO mode actions are allowed
        for action in actions:
            local_time = action.timestamp.astimezone()
            hour = local_time.hour
            in_quiet_window = 9 <= hour < 17

            # If in quiet window and setting temperature, should not happen
            if in_quiet_window and action.temperature is not None:
                assert False, f"Temperature-setting action at {local_time} should be filtered in quiet window"

    def test_calculate_heating_schedule_with_quiet_window_midnight_crossing(self) -> None:
        """Test that temperature changes during a midnight-crossing quiet window are filtered out, but ECO mode changes allowed."""
        # Create prices that would generate actions during 23:00-07:00
        prices = []
        base_time = datetime(2024, 1, 1, 20, 0, tzinfo=timezone.utc)

        # Create alternating low/high prices
        for i in range(48):  # 24 hours
            hour_offset = i // 2
            minute = (i % 2) * 30
            valid_from = base_time + timedelta(hours=hour_offset, minutes=minute)
            valid_to = valid_from + timedelta(minutes=30)

            # Alternate between low (3p) and high (16p) prices every 2 hours
            price = 3.0 if (i // 4) % 2 == 0 else 16.0

            prices.append(create_price_point(
                valid_from.isoformat(),
                valid_to.isoformat(),
                price
            ))

        weekly_prices = [create_price_point(
            '2023-12-25T00:00:00Z',
            '2023-12-25T00:30:00Z',
            10.0
        )] * 336

        config = Config(
            thermostat_name='Test',
            client_id='test',
            client_secret='test',
            refresh_token='test',
            project_id='test',
            tariff_code='AGILE-TEST',
            quiet_window=(23, 0, 7, 0)  # 23:00-07:00 (crosses midnight)
        )

        actions = calculate_heating_schedule(prices, weekly_prices, config, base_time)

        # Check that no temperature-setting actions fall within 23:00-07:00 local time
        # But ECO mode actions (temperature=None or eco_mode=True) are allowed
        for action in actions:
            local_time = action.timestamp.astimezone()
            hour = local_time.hour
            in_quiet_window = 23 <= hour or hour < 7

            # If in quiet window and setting temperature, should not happen
            if in_quiet_window and action.temperature is not None:
                assert False, f"Temperature-setting action at {local_time} should be filtered in quiet window"

    def test_calculate_heating_schedule_without_quiet_window(self) -> None:
        """Test that quiet window filters temperature-setting actions but not ECO mode changes."""
        # Create scenario where temperature changes between tier (3p) and default (12p)
        prices = []
        base_time = datetime(2024, 1, 1, 20, 0, tzinfo=timezone.utc)

        for i in range(48):
            hour_offset = i // 2
            minute = (i % 2) * 30
            valid_from = base_time + timedelta(hours=hour_offset, minutes=minute)
            valid_to = valid_from + timedelta(minutes=30)
            # Alternate every 2 hours: 20:00-22:00 @ 3p, 22:00-00:00 @ 12p, 00:00-02:00 @ 3p, etc.
            price = 3.0 if (hour_offset % 4) < 2 else 12.0

            prices.append(create_price_point(
                valid_from.isoformat(),
                valid_to.isoformat(),
                price
            ))

        weekly_prices = [create_price_point(
            '2023-12-25T00:00:00Z',
            '2023-12-25T00:30:00Z',
            10.0
        )] * 336

        # Without quiet window
        config_no_quiet = Config(
            thermostat_name='Test',
            client_id='test',
            client_secret='test',
            refresh_token='test',
            project_id='test',
            tariff_code='AGILE-TEST',
            quiet_window=None,
            temperature_tiers=[TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)],
            default_temp=17.0
        )

        # With quiet window (23:00-07:00)
        config_with_quiet = Config(
            thermostat_name='Test',
            client_id='test',
            client_secret='test',
            refresh_token='test',
            project_id='test',
            tariff_code='AGILE-TEST',
            quiet_window=(23, 0, 7, 0),
            temperature_tiers=[TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)],
            default_temp=17.0
        )

        actions_no_quiet = calculate_heating_schedule(prices, weekly_prices, config_no_quiet, base_time)
        actions_with_quiet = calculate_heating_schedule(prices, weekly_prices, config_with_quiet, base_time)

        # Count temperature-setting actions in quiet window (23:00-07:00) without quiet window config
        temp_actions_in_quiet = [a for a in actions_no_quiet
                                 if a.temperature is not None
                                 and ((a.timestamp.hour >= 23) or (a.timestamp.hour < 7))]

        # Should have fewer actions with quiet window (temperature changes filtered)
        assert len(temp_actions_in_quiet) > 0, "Should have temperature actions in quiet hours without config"
        assert len(actions_with_quiet) < len(actions_no_quiet)

    def test_quiet_window_allows_eco_mode_blocks_temperature(self) -> None:
        """Test that quiet window blocks temperature changes but allows ECO mode changes."""
        # Create prices that alternate to generate both temp and ECO actions
        prices = []
        base_time = datetime(2024, 1, 1, 20, 0, tzinfo=timezone.utc)

        for i in range(48):
            hour_offset = i // 2
            minute = (i % 2) * 30
            valid_from = base_time + timedelta(hours=hour_offset, minutes=minute)
            valid_to = valid_from + timedelta(minutes=30)

            # Alternate between low (3p) and high (16p) every 2 hours
            price = 3.0 if (i // 4) % 2 == 0 else 16.0

            prices.append(create_price_point(
                valid_from.isoformat(),
                valid_to.isoformat(),
                price
            ))

        weekly_prices = [create_price_point(
            '2023-12-25T00:00:00Z',
            '2023-12-25T00:30:00Z',
            10.0
        )] * 336

        config = Config(
            thermostat_name='Test',
            client_id='test',
            client_secret='test',
            refresh_token='test',
            project_id='test',
            tariff_code='AGILE-TEST',
            quiet_window=(23, 0, 7, 0)
        )

        actions = calculate_heating_schedule(prices, weekly_prices, config, base_time)

        # Count ECO mode actions and temperature actions in quiet window
        eco_actions_in_quiet = 0
        temp_actions_in_quiet = 0

        for action in actions:
            local_time = action.timestamp.astimezone()
            hour = local_time.hour
            in_quiet_window = 23 <= hour or hour < 7

            if in_quiet_window:
                if action.eco_mode or action.temperature is None:
                    # ECO mode change (enable or disable)
                    eco_actions_in_quiet += 1
                elif action.temperature is not None:
                    # Temperature-setting action
                    temp_actions_in_quiet += 1

        # Should have ECO mode actions in quiet window
        assert eco_actions_in_quiet > 0, "Should have ECO mode actions during quiet window"
        # Should NOT have temperature-setting actions in quiet window
        assert temp_actions_in_quiet == 0, "Should not have temperature-setting actions during quiet window"


class TestTGActivePeriod:
    """Test TG SupplyMaster active_period functionality."""

    def test_parse_tg_active_period_normal(self) -> None:
        """Test parsing normal active period."""
        result = parse_tg_active_period("05:00-20:00")
        assert result == (5, 0, 20, 0)

    def test_parse_tg_active_period_midnight_crossing(self) -> None:
        """Test parsing midnight-crossing active period."""
        result = parse_tg_active_period("22:00-06:00")
        assert result == (22, 0, 6, 0)

    def test_parse_tg_active_period_invalid(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid TG active period format"):
            parse_tg_active_period("invalid")

    def test_find_cheapest_windows_with_active_period_excludes_boundary_violation(self) -> None:
        """Test that windows ending outside active_period are excluded."""
        # Create prices where 19:30-21:30 would be cheap but crosses 20:00 boundary
        prices = []
        for hour in range(24):
            if hour in [5, 6]:
                price = 13.0  # Cheap at 5-7am
            elif hour in [19, 20]:
                price = 14.0  # Cheaper at 7-9pm but crosses boundary
            elif hour in [18]:
                price = 15.0  # Slightly more expensive but fits
            else:
                price = 25.0  # Expensive

            prices.append(create_price_point(
                f'2024-12-08T{hour:02d}:00:00Z',
                f'2024-12-08T{hour:02d}:30:00Z',
                price
            ))
            if hour < 23:
                prices.append(create_price_point(
                    f'2024-12-08T{hour:02d}:30:00Z',
                    f'2024-12-08T{(hour+1):02d}:00:00Z',
                    price
                ))

        # Find windows with active_period = 05:00-20:00
        windows = find_cheapest_windows(
            prices,
            window_hours=2,
            num_windows=2,
            min_gap_hours=4,
            active_period=(5, 0, 20, 0)
        )

        # Should find 2 windows
        assert len(windows) == 2, f"Expected 2 windows, got {len(windows)}"

        # First window should be 05:00-07:00
        start1, end1, _ = windows[0]
        assert start1.astimezone().hour == 5 and start1.astimezone().minute == 0
        assert end1.astimezone().hour == 7 and end1.astimezone().minute == 0

        # Second window should be 18:00-20:00, NOT 19:30-21:30
        start2, end2, _ = windows[1]
        assert start2.astimezone().hour == 18 and start2.astimezone().minute == 0
        assert end2.astimezone().hour == 20 and end2.astimezone().minute == 0

        # Verify no window ends after 20:00
        for start, end, _ in windows:
            end_local = end.astimezone()
            end_minutes = end_local.hour * 60 + end_local.minute
            assert end_minutes <= 20 * 60, f"Window {start.strftime('%H:%M')}-{end.strftime('%H:%M')} ends after 20:00"

    def test_find_cheapest_windows_with_active_period_normal_period(self) -> None:
        """Test active_period with normal daytime period."""
        prices = []
        for hour in range(24):
            if hour in [2, 3]:
                price = 10.0  # Very cheap at 2-4am (excluded)
            elif hour in [5, 6]:
                price = 13.0  # Cheap at 5-7am (included)
            else:
                price = 25.0

            prices.append(create_price_point(
                f'2024-12-08T{hour:02d}:00:00Z',
                f'2024-12-08T{hour:02d}:30:00Z',
                price
            ))
            if hour < 23:
                prices.append(create_price_point(
                    f'2024-12-08T{hour:02d}:30:00Z',
                    f'2024-12-08T{(hour+1):02d}:00:00Z',
                    price
                ))

        windows = find_cheapest_windows(
            prices,
            window_hours=2,
            num_windows=1,
            min_gap_hours=4,
            active_period=(5, 0, 20, 0)
        )

        # Should find 05:00-07:00, not 02:00-04:00
        assert len(windows) == 1
        start, end, _ = windows[0]
        assert start.astimezone().hour == 5 and start.astimezone().minute == 0
        assert end.astimezone().hour == 7 and end.astimezone().minute == 0

    def test_find_cheapest_windows_with_active_period_midnight_crossing(self) -> None:
        """Test active_period with midnight-crossing period."""
        prices = []
        for hour in range(24):
            if hour in [2, 3]:
                price = 10.0  # Very cheap at 2-4am (included)
            elif hour in [5, 6]:
                price = 13.0  # Cheap at 5-7am (excluded - ends after 6am)
            elif hour in [22, 23]:
                price = 12.0  # Cheap at 22-00 (included)
            else:
                price = 25.0

            prices.append(create_price_point(
                f'2024-12-08T{hour:02d}:00:00Z',
                f'2024-12-08T{hour:02d}:30:00Z',
                price
            ))
            if hour < 23:
                prices.append(create_price_point(
                    f'2024-12-08T{hour:02d}:30:00Z',
                    f'2024-12-08T{(hour+1):02d}:00:00Z',
                    price
                ))

        windows = find_cheapest_windows(
            prices,
            window_hours=2,
            num_windows=2,
            min_gap_hours=4,
            active_period=(22, 0, 6, 0)  # Night only: 22:00-06:00
        )

        # Should find 02:00-04:00 and/or 22:00-00:00
        assert len(windows) >= 1

        # Check that all windows are within the midnight-crossing period
        for start, end, _ in windows:
            start_local = start.astimezone()
            end_local = end.astimezone()
            start_hour = start_local.hour
            end_hour = end_local.hour

            # Start should be >= 22:00 or < 06:00
            assert start_hour >= 22 or start_hour < 6, \
                f"Window start {start_hour}:00 not in 22:00-06:00 period"

            # End should be >= 22:00 or <= 06:00
            assert end_hour >= 22 or end_hour <= 6, \
                f"Window end {end_hour}:00 not in 22:00-06:00 period"

    def test_find_cheapest_windows_without_active_period(self) -> None:
        """Test that windows work normally without active_period."""
        prices = []
        for hour in range(24):
            if hour in [2, 3]:
                price = 10.0  # Very cheap at 2-4am
            else:
                price = 25.0

            prices.append(create_price_point(
                f'2024-12-08T{hour:02d}:00:00Z',
                f'2024-12-08T{hour:02d}:30:00Z',
                price
            ))
            if hour < 23:
                prices.append(create_price_point(
                    f'2024-12-08T{hour:02d}:30:00Z',
                    f'2024-12-08T{(hour+1):02d}:00:00Z',
                    price
                ))

        windows = find_cheapest_windows(
            prices,
            window_hours=2,
            num_windows=1,
            min_gap_hours=4,
            active_period=None  # No restriction
        )

        # Should find cheapest window regardless of time
        assert len(windows) == 1
        start, end, _ = windows[0]
        assert start.astimezone().hour == 2 and start.astimezone().minute == 0
        assert end.astimezone().hour == 4 and end.astimezone().minute == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


