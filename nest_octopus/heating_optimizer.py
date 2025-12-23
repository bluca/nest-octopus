# SPDX-License-Identifier: MPL-2.0
"""
Heating Optimization Daemon

Queries electricity prices daily at 10pm and optimizes heating schedule to minimize costs
while maintaining comfort. Uses dynamic pricing to determine when to heat (22°C),
maintain comfort (17°C), or enable ECO mode during high prices.

The daemon:
1. Fetches electricity prices for the next 24 hours at 10pm
2. Analyzes prices against daily and weekly averages
3. Calculates optimal heating schedule
4. Executes temperature/mode changes at calculated times
5. Sleeps between changes until next action
6. Repeats cycle daily at 10pm
"""

import argparse
import asyncio
import configparser
import errno
import logging
import os
import signal
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from nest_octopus.nest_thermostat import EcoMode, NestThermostatClient, ThermostatMode
from nest_octopus.ntfy import NtfyClient
from nest_octopus.octopus import OctopusEnergyClient, PricePoint
from nest_octopus.tg_supplymaster import (
    DayOfWeek,
    Program,
    ProgramSlot,
    SupplyMasterClient,
    TimeSlot,
    WorkMode,
)

# Logger will be configured later based on config/CLI args
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


@dataclass(frozen=True)
class TimeRange:
    """
    Represents a time range within a day.

    Supports ranges that cross midnight (e.g., 23:00-07:00).
    Uses Python's standard library datetime.time for type safety.
    """
    start: dt_time
    end: dt_time

    def contains(self, t: dt_time) -> bool:
        """
        Check if a time falls within this range.

        Handles ranges that cross midnight correctly.
        Note: The end time is exclusive.

        Args:
            t: The time to check

        Returns:
            True if the time is within the range, False otherwise
        """
        # Convert to minutes for comparison
        t_minutes = t.hour * 60 + t.minute
        start_minutes = self.start.hour * 60 + self.start.minute
        end_minutes = self.end.hour * 60 + self.end.minute

        if start_minutes <= end_minutes:
            # Normal range (e.g., 09:00-17:00)
            return start_minutes <= t_minutes < end_minutes
        else:
            # Range crosses midnight (e.g., 23:00-07:00)
            return t_minutes >= start_minutes or t_minutes < end_minutes

    def contains_window(self, window_start: dt_time, window_end: dt_time) -> bool:
        """
        Check if a time window is fully contained within this range.

        This is used for checking if a scheduled window fits within an active period.
        The window_end is treated as inclusive (i.e., a window ending exactly at
        the active period end time is allowed).

        Args:
            window_start: Start time of the window to check
            window_end: End time of the window to check

        Returns:
            True if the entire window is within this range, False otherwise
        """
        window_start_minutes = window_start.hour * 60 + window_start.minute
        window_end_minutes = window_end.hour * 60 + window_end.minute
        start_minutes = self.start.hour * 60 + self.start.minute
        end_minutes = self.end.hour * 60 + self.end.minute

        if end_minutes < start_minutes:
            # Period crosses midnight (e.g., 22:00-06:00)
            start_in_period = (window_start_minutes >= start_minutes or
                              window_start_minutes < end_minutes)
            end_in_period = (window_end_minutes >= start_minutes or
                            window_end_minutes <= end_minutes)
            return start_in_period and end_in_period
        else:
            # Normal period (e.g., 05:00-20:00)
            start_in_period = (start_minutes <= window_start_minutes < end_minutes)
            end_in_period = (start_minutes < window_end_minutes <= end_minutes)
            return start_in_period and end_in_period

    def __str__(self) -> str:
        """Return string representation in hh:mm-hh:mm format."""
        return f"{self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')}"


def parse_time_range(value: str, param_name: str = "time range") -> TimeRange:
    """
    Parse time range from format hh:mm-hh:mm.

    Args:
        value: String in format "hh:mm-hh:mm" (e.g., "23:00-07:00")
        param_name: Name of parameter for error messages

    Returns:
        TimeRange object with start and end times

    Raises:
        ValueError: If format is invalid
    """
    if '-' not in value:
        raise ValueError(f"Invalid {param_name} format: '{value}'. Expected 'hh:mm-hh:mm'")

    try:
        start_str, end_str = value.split('-', 1)
        start_parts = start_str.strip().split(':')
        end_parts = end_str.strip().split(':')

        if len(start_parts) != 2 or len(end_parts) != 2:
            raise ValueError("Times must be in hh:mm format")

        start_hour = int(start_parts[0])
        start_min = int(start_parts[1])
        end_hour = int(end_parts[0])
        end_min = int(end_parts[1])

        # Validate ranges
        if not (0 <= start_hour <= 23):
            raise ValueError(f"Start hour must be 0-23, got {start_hour}")
        if not (0 <= start_min <= 59):
            raise ValueError(f"Start minute must be 0-59, got {start_min}")
        if not (0 <= end_hour <= 23):
            raise ValueError(f"End hour must be 0-23, got {end_hour}")
        if not (0 <= end_min <= 59):
            raise ValueError(f"End minute must be 0-59, got {end_min}")

        return TimeRange(dt_time(start_hour, start_min), dt_time(end_hour, end_min))

    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid {param_name} format: '{value}'. {e}")


def parse_quiet_window(value: str) -> TimeRange:
    """
    Parse quiet window time range from format hh:mm-hh:mm.

    Args:
        value: String in format "hh:mm-hh:mm" (e.g., "23:00-07:00")

    Returns:
        TimeRange object

    Raises:
        ValueError: If format is invalid
    """
    return parse_time_range(value, "quiet window")


def parse_tg_active_period(value: str) -> TimeRange:
    """
    Parse TG active period time range from format hh:mm-hh:mm.

    Args:
        value: String in format "hh:mm-hh:mm" (e.g., "04:00-20:00")

    Returns:
        TimeRange object

    Raises:
        ValueError: If format is invalid
    """
    return parse_time_range(value, "TG active period")


def parse_cycle_time(value: str) -> dt_time:
    """
    Parse cycle time from format hh:mm.

    Args:
        value: String in format "hh:mm" (e.g., "21:50")

    Returns:
        datetime.time object

    Raises:
        ValueError: If format is invalid
    """
    value = value.strip()

    if ':' not in value:
        raise ValueError(
            f"Invalid cycle time format: '{value}'. "
            f"Expected 'hh:mm' format. Example: '21:50'"
        )

    try:
        parts = value.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid cycle time format: '{value}'")

        hour = int(parts[0])
        minute = int(parts[1])

        if not (0 <= hour <= 23):
            raise ValueError(f"Hour must be 0-23, got {hour}")
        if not (0 <= minute <= 59):
            raise ValueError(f"Minute must be 0-59, got {minute}")

        return dt_time(hour, minute)

    except (ValueError, IndexError) as e:
        if "Hour must be" in str(e) or "Minute must be" in str(e):
            raise
        raise ValueError(
            f"Invalid cycle time format: '{value}'. "
            f"Expected 'hh:mm' format. Example: '21:50'. Error: {e}"
        )


@dataclass
class TemperatureTier:
    """Represents a temperature tier with its price threshold."""
    temperature: float  # Temperature in °C
    threshold_pct: Optional[float] = None  # Percentage multiplier (e.g., 0.75 for 75%)
    threshold_abs: Optional[float] = None  # Absolute price in pence

    def __post_init__(self) -> None:
        """Validate that at least one threshold is set."""
        if self.threshold_pct is None and self.threshold_abs is None:
            raise ValueError("At least one of threshold_pct or threshold_abs must be set")


@dataclass
class NotifyThreshold:
    """
    Represents a price threshold for sending notifications.

    Either threshold_pct (compared to weekly average) or threshold_abs must be set.
    """
    threshold_pct: Optional[float] = None  # Percentage multiplier (e.g., 1.20 for 120%)
    threshold_abs: Optional[float] = None  # Absolute price in pence

    def __post_init__(self) -> None:
        """Validate that exactly one threshold type is set."""
        if self.threshold_pct is None and self.threshold_abs is None:
            raise ValueError("One of threshold_pct or threshold_abs must be set")
        if self.threshold_pct is not None and self.threshold_abs is not None:
            raise ValueError("Only one of threshold_pct or threshold_abs can be set")

    def __str__(self) -> str:
        """Return string representation."""
        if self.threshold_pct is not None:
            return f"{self.threshold_pct * 100:.0f}%"
        return f"{self.threshold_abs}p"


class PeriodDict(TypedDict):
    """Type definition for heating period dictionary."""
    temperature: Optional[float]
    eco_mode: bool
    start: datetime
    end: datetime


def parse_temperature_tier(value: str) -> TemperatureTier:
    """
    Parse temperature tier in format <temp>@<price>.

    Args:
        value: String in format "<temp>@<price>" where:
               - temp is temperature in °C (e.g., "22", "20.5")
               - price ends with '%' (percentage) or 'p' (pence)
               Examples: "22@50%", "20@15p", "17.5@75%"

    Returns:
        TemperatureTier object

    Raises:
        ValueError: If format is invalid
    """
    value = value.strip()

    if '@' not in value:
        raise ValueError(
            f"Invalid temperature tier format: '{value}'. "
            f"Expected '<temp>@<price>'. Examples: '22@50%', '20@15p'"
        )

    try:
        temp_str, price_str = value.split('@', 1)
        temperature = float(temp_str.strip())
        price_str = price_str.strip()

        # Parse price threshold
        if price_str.endswith('%'):
            percentage = float(price_str[:-1])
            # Convert percentage to multiplier (e.g., 75% -> 0.75)
            return TemperatureTier(
                temperature=temperature,
                threshold_pct=percentage / 100.0,
                threshold_abs=None
            )
        elif price_str.endswith('p'):
            pence = float(price_str[:-1])
            if pence < 0:
                raise ValueError("Price must be non-negative")
            return TemperatureTier(
                temperature=temperature,
                threshold_pct=None,
                threshold_abs=pence
            )
        else:
            raise ValueError(
                f"Invalid price format: '{price_str}'. "
                f"Must end with '%' (percentage) or 'p' (pence)"
            )

    except (ValueError, IndexError) as e:
        if isinstance(e, ValueError) and "Price must be non-negative" in str(e):
            raise
        if isinstance(e, ValueError) and "Invalid price format" in str(e):
            raise
        raise ValueError(
            f"Invalid temperature tier format: '{value}'. "
            f"Expected '<temp>@<price>'. Examples: '22@50%', '20@15p'. Error: {e}"
        )


def parse_notify_threshold(value: str, param_name: str = "threshold") -> NotifyThreshold:
    """
    Parse notification threshold from format '<value>%' or '<value>p'.

    Args:
        value: String ending with '%' (percentage) or 'p' (pence).
               Examples: "120%", "15.5p", "80%"
        param_name: Name of parameter for error messages

    Returns:
        NotifyThreshold object

    Raises:
        ValueError: If format is invalid
    """
    value = value.strip()

    if not value:
        raise ValueError(f"Empty {param_name} value")

    try:
        if value.endswith('%'):
            percentage = float(value[:-1])
            return NotifyThreshold(
                threshold_pct=percentage / 100.0,
                threshold_abs=None
            )
        elif value.endswith('p'):
            pence = float(value[:-1])
            if pence < 0:
                raise ValueError(f"{param_name} must be non-negative")
            return NotifyThreshold(
                threshold_pct=None,
                threshold_abs=pence
            )
        else:
            raise ValueError(
                f"Invalid {param_name} format: '{value}'. "
                f"Must end with '%' (percentage of weekly avg) or 'p' (pence)"
            )
    except ValueError as e:
        if "non-negative" in str(e) or "Invalid" in str(e):
            raise
        raise ValueError(
            f"Invalid {param_name} format: '{value}'. "
            f"Examples: '120%', '15.5p'. Error: {e}"
        )


@dataclass
class Config:
    """Application configuration."""
    # Nest Thermostat
    thermostat_name: str
    client_id: str
    client_secret: str
    refresh_token: str
    project_id: str

    # Optional Octopus Energy settings
    tariff_code: Optional[str] = None
    account_number: Optional[str] = None
    api_key: Optional[str] = None
    mpan: Optional[str] = None

    # Heating preferences
    temperature_tiers: List[TemperatureTier] = field(default_factory=lambda: [
        TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)
    ])  # Temperature tiers sorted by threshold
    default_temp: float = 17.0  # Default temperature when no tier matches
    eco_threshold_pct: Optional[float] = 1.33  # Percentage threshold for ECO mode
    eco_threshold_abs: Optional[float] = None  # Absolute price threshold for ECO mode
    quiet_window: Optional[TimeRange] = None  # Time range to avoid temperature changes

    # Optional TG SupplyMaster settings
    tg_username: Optional[str] = None
    tg_password: Optional[str] = None
    tg_device_name: Optional[str] = None
    tg_window_hours: int = 2  # Duration of each window
    tg_num_windows: int = 2  # Number of windows per day
    tg_min_gap_hours: int = 10  # Minimum gap between windows
    tg_active_period: Optional[TimeRange] = None  # Time range to restrict windows

    # Logging
    logging_level: str = 'WARNING'  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    # Daily cycle time
    cycle_time: dt_time = field(default_factory=lambda: dt_time(21, 50))  # Time for daily cycle

    # Optional ntfy notification settings
    ntfy_topic: Optional[str] = None  # Topic for ntfy notifications
    ntfy_server: str = "https://ntfy.sh"  # ntfy server URL
    ntfy_token: Optional[str] = None  # Access token for ntfy authentication
    ntfy_high_threshold: Optional[NotifyThreshold] = None  # Notify when price goes above
    ntfy_low_threshold: Optional[NotifyThreshold] = None  # Notify when price goes below


@dataclass
class HeatingAction:
    """Represents a scheduled heating change."""
    timestamp: datetime
    temperature: Optional[float]
    eco_mode: bool
    reason: str

    def __repr__(self) -> str:
        if self.eco_mode:
            return f"HeatingAction({self.timestamp.strftime('%H:%M')}, ECO, {self.reason})"
        return f"HeatingAction({self.timestamp.strftime('%H:%M')}, {self.temperature}°C, {self.reason})"


def notify(message: bytes) -> None:
    if not message:
        raise ValueError("notify() requires a message")

    socket_path = os.environ.get("NOTIFY_SOCKET")
    if not socket_path:
        return

    if socket_path[0] not in ("/", "@"):
        raise OSError(errno.EAFNOSUPPORT, "Unsupported socket type")

    # Handle abstract socket.
    if socket_path[0] == "@":
        socket_path = "\0" + socket_path[1:]

    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as sock:
        sock.connect(socket_path)
        sock.sendall(message)


def notify_ready() -> None:
    notify(b"READY=1")


def notify_reloading() -> None:
    microsecs = time.clock_gettime_ns(time.CLOCK_MONOTONIC) // 1000
    notify(f"RELOADING=1\nMONOTONIC_USEC={microsecs}".encode())


def notify_stopping() -> None:
    notify(b"STOPPING=1")


def find_default_config() -> Optional[str]:
    """
    Find configuration file using standard search paths.

    Search order:
    1. /etc/nest-octopus/nest-octopus.conf
    2. /run/nest-octopus/nest-octopus.conf
    3. /usr/lib/nest-octopus/nest-octopus.conf

    Returns:
        Path to first existing config file, or None if none found
    """
    search_paths = [
        "/etc/nest-octopus/nest-octopus.conf",
        "/run/nest-octopus/nest-octopus.conf",
        "/usr/lib/nest-octopus/nest-octopus.conf",
    ]

    for path in search_paths:
        if os.path.exists(path):
            logger.debug(f"Found configuration file: {path}")
            return path

    return None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from INI file and credentials directory.

    Args:
        config_path: Path to configuration INI file. If None, searches default locations.

    Returns:
        Config object with all settings

    Raises:
        ConfigurationError: If configuration is invalid or credentials missing
    """
    # If no path specified, search default locations
    if config_path is None:
        config_path = find_default_config()
        if config_path is None:
            raise ConfigurationError(
                "No configuration file found. Searched: "
                "/etc/nest-octopus/nest-octopus.conf, "
                "/run/nest-octopus/nest-octopus.conf, "
                "/usr/lib/nest-octopus/nest-octopus.conf"
            )

    if not os.path.exists(config_path):
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config_path)

    # Get credentials directory from environment
    creds_dir = os.getenv('CREDENTIALS_DIRECTORY')
    if not creds_dir:
        raise ConfigurationError(
            "CREDENTIALS_DIRECTORY environment variable not set"
        )

    creds_path = Path(creds_dir)
    if not creds_path.exists():
        raise ConfigurationError(
            f"Credentials directory does not exist: {creds_dir}"
        )

    # Read credentials from files
    client_secret_file = creds_path / "client_secret"
    refresh_token_file = creds_path / "refresh_token"
    api_key_file = creds_path / "api_key"
    tg_password_file = creds_path / "tg_password"

    if not client_secret_file.exists():
        raise ConfigurationError(
            f"client_secret file not found in {creds_dir}"
        )
    if not refresh_token_file.exists():
        raise ConfigurationError(
            f"refresh_token file not found in {creds_dir}"
        )

    client_secret = client_secret_file.read_text().strip()
    refresh_token = refresh_token_file.read_text().strip()

    # API key is optional
    api_key = None
    if api_key_file.exists():
        api_key = api_key_file.read_text().strip()

    # TG password is optional
    tg_password = None
    if tg_password_file.exists():
        tg_password = tg_password_file.read_text().strip()

    # Parse configuration sections
    try:
        # Tariff code and account number are optional in config
        tariff_code = None
        if parser.has_option('octopus', 'tariff_code'):
            tariff_code = parser.get('octopus', 'tariff_code')

        account_number = None
        if parser.has_option('octopus', 'account_number'):
            account_number = parser.get('octopus', 'account_number')

        mpan = None
        if parser.has_option('octopus', 'mpan'):
            mpan = parser.get('octopus', 'mpan')

        # Optional TG SupplyMaster settings
        tg_username = None
        if parser.has_section('tg_supplymaster') and parser.has_option('tg_supplymaster', 'username'):
            tg_username = parser.get('tg_supplymaster', 'username')

        tg_device_name = None
        if parser.has_section('tg_supplymaster') and parser.has_option('tg_supplymaster', 'device_name'):
            tg_device_name = parser.get('tg_supplymaster', 'device_name')

        tg_window_hours = 2
        if parser.has_section('tg_supplymaster') and parser.has_option('tg_supplymaster', 'window_hours'):
            tg_window_hours = parser.getint('tg_supplymaster', 'window_hours')

        tg_num_windows = 2
        if parser.has_section('tg_supplymaster') and parser.has_option('tg_supplymaster', 'num_windows'):
            tg_num_windows = parser.getint('tg_supplymaster', 'num_windows')

        tg_min_gap_hours = 10
        if parser.has_section('tg_supplymaster') and parser.has_option('tg_supplymaster', 'min_gap_hours'):
            tg_min_gap_hours = parser.getint('tg_supplymaster', 'min_gap_hours')

        tg_active_period = None
        if parser.has_section('tg_supplymaster') and parser.has_option('tg_supplymaster', 'active_period'):
            tg_active_period = parse_tg_active_period(parser.get('tg_supplymaster', 'active_period'))

        config = Config(
            thermostat_name=parser.get('nest', 'thermostat_name'),
            client_id=parser.get('nest', 'client_id'),
            project_id=parser.get('nest', 'project_id'),
            client_secret=client_secret,
            refresh_token=refresh_token,
            tariff_code=tariff_code,
            account_number=account_number,
            api_key=api_key,
            mpan=mpan,
            tg_username=tg_username,
            tg_password=tg_password,
            tg_device_name=tg_device_name,
            tg_window_hours=tg_window_hours,
            tg_num_windows=tg_num_windows,
            tg_min_gap_hours=tg_min_gap_hours,
            tg_active_period=tg_active_period,
        )

        # Optional heating preferences
        temperature_tiers = []
        default_temp = 17.0
        eco_threshold_pct = 1.33
        eco_threshold_abs = None

        # Parse temperature tiers from price_threshold option
        if parser.has_section('heating'):
            if parser.has_option('heating', 'price_threshold'):
                # Get the option value which can have multiple space-separated tiers
                threshold_str = parser.get('heating', 'price_threshold')
                # Split by whitespace and parse each tier
                for tier_str in threshold_str.split():
                    tier_str = tier_str.strip()
                    if tier_str:  # Skip empty strings
                        temperature_tiers.append(parse_temperature_tier(tier_str))

            # Parse default temperature
            if parser.has_option('heating', 'default_temp'):
                default_temp = parser.getfloat('heating', 'default_temp')

            # Parse ECO threshold
            if parser.has_option('heating', 'eco_threshold'):
                eco_str = parser.get('heating', 'eco_threshold').strip()
                if eco_str.endswith('%'):
                    eco_threshold_pct = float(eco_str[:-1]) / 100.0
                    eco_threshold_abs = None
                elif eco_str.endswith('p'):
                    eco_threshold_abs = float(eco_str[:-1])
                    eco_threshold_pct = 1.33  # Keep a default even when using absolute

        # Set parsed values on config
        if temperature_tiers:
            config.temperature_tiers = temperature_tiers
        config.default_temp = default_temp
        if eco_threshold_pct is not None:
            config.eco_threshold_pct = eco_threshold_pct
        if eco_threshold_abs is not None:
            config.eco_threshold_abs = eco_threshold_abs

        if parser.has_option('heating', 'quiet_window'):
            config.quiet_window = parse_quiet_window(parser.get('heating', 'quiet_window'))

        if parser.has_option('heating', 'cycle_time'):
            config.cycle_time = parse_cycle_time(parser.get('heating', 'cycle_time'))

        # Optional ntfy notification settings
        if parser.has_section('ntfy'):
            if parser.has_option('ntfy', 'topic'):
                config.ntfy_topic = parser.get('ntfy', 'topic')

            if parser.has_option('ntfy', 'server'):
                config.ntfy_server = parser.get('ntfy', 'server')

            if parser.has_option('ntfy', 'high_threshold'):
                config.ntfy_high_threshold = parse_notify_threshold(
                    parser.get('ntfy', 'high_threshold'),
                    "ntfy high_threshold"
                )

            if parser.has_option('ntfy', 'low_threshold'):
                config.ntfy_low_threshold = parse_notify_threshold(
                    parser.get('ntfy', 'low_threshold'),
                    "ntfy low_threshold"
                )

        # Read ntfy token from credentials directory (optional)
        ntfy_token_file = creds_path / "ntfy_token"
        if ntfy_token_file.exists():
            config.ntfy_token = ntfy_token_file.read_text().strip()

        # Optional logging configuration
        if parser.has_option('logging', 'level'):
            config.logging_level = parser.get('logging', 'level').upper()

        # Validate Octopus Energy configuration
        if not config.tariff_code and not (config.api_key and config.account_number):
            raise ConfigurationError(
                "Either tariff_code or both api_key and account_number must be configured"
            )

        return config

    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise ConfigurationError(f"Invalid configuration: {e}")


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    """
    Apply command line argument overrides to configuration.

    Args:
        config: Configuration object to modify
        args: Parsed command line arguments
    """
    if args.tariff_code:
        logger.debug(f"Overriding tariff code with: {args.tariff_code}")
        config.tariff_code = args.tariff_code

    if hasattr(args, 'price_threshold') and args.price_threshold:
        logger.debug(f"Adding {len(args.price_threshold)} price threshold(s)")
        config.temperature_tiers = args.price_threshold

    if hasattr(args, 'default_temp') and args.default_temp is not None:
        config.default_temp = args.default_temp

    if hasattr(args, 'eco_threshold') and args.eco_threshold is not None:
        eco_str = args.eco_threshold.strip()
        if eco_str.endswith('%'):
            config.eco_threshold_pct = float(eco_str[:-1]) / 100.0
            config.eco_threshold_abs = None
        elif eco_str.endswith('p'):
            config.eco_threshold_abs = float(eco_str[:-1])
            config.eco_threshold_pct = None

    if args.tg_username is not None:
        config.tg_username = args.tg_username

    if args.tg_password is not None:
        config.tg_password = args.tg_password

    if args.tg_device_name is not None:
        config.tg_device_name = args.tg_device_name

    if args.tg_window_hours is not None:
        config.tg_window_hours = args.tg_window_hours

    if args.tg_num_windows is not None:
        config.tg_num_windows = args.tg_num_windows

    if args.tg_min_gap_hours is not None:
        config.tg_min_gap_hours = args.tg_min_gap_hours

    if args.tg_active_period is not None:
        config.tg_active_period = args.tg_active_period

    if args.quiet_window is not None:
        config.quiet_window = args.quiet_window

    if args.cycle_time is not None:
        config.cycle_time = args.cycle_time

    if args.log_level is not None:
        config.logging_level = args.log_level.upper()

    # ntfy notification overrides
    if hasattr(args, 'ntfy_topic') and args.ntfy_topic is not None:
        config.ntfy_topic = args.ntfy_topic

    if hasattr(args, 'ntfy_server') and args.ntfy_server is not None:
        config.ntfy_server = args.ntfy_server

    if hasattr(args, 'ntfy_token') and args.ntfy_token is not None:
        config.ntfy_token = args.ntfy_token

    if hasattr(args, 'ntfy_high_threshold') and args.ntfy_high_threshold is not None:
        config.ntfy_high_threshold = args.ntfy_high_threshold

    if hasattr(args, 'ntfy_low_threshold') and args.ntfy_low_threshold is not None:
        config.ntfy_low_threshold = args.ntfy_low_threshold


def configure_logging(level: str) -> None:
    """
    Configure logging level for all modules.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Map string to logging level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    log_level = level_map.get(level.upper(), logging.WARNING)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(name)s - %(levelname)s - %(message)s',
        force=True,
    )

    # Configure module loggers
    logging.getLogger('nest_octopus.heating_optimizer').setLevel(log_level)
    logging.getLogger('nest_octopus.nest_thermostat').setLevel(log_level)
    logging.getLogger('nest_octopus.octopus').setLevel(log_level)
    logging.getLogger('nest_octopus.tg_supplymaster').setLevel(log_level)


def calculate_price_statistics(
    daily_prices: List[PricePoint],
    weekly_prices: List[PricePoint]
) -> Tuple[float, float, float, float]:
    """
    Calculate price statistics for decision making.

    Args:
        daily_prices: Prices for the next 24 hours
        weekly_prices: Prices for the preceding week

    Returns:
        Tuple of (daily_avg, weekly_avg, daily_min, daily_max)
    """
    daily_values = [p.value_inc_vat for p in daily_prices]
    weekly_values = [p.value_inc_vat for p in weekly_prices]

    daily_avg = sum(daily_values) / len(daily_values)
    weekly_avg = sum(weekly_values) / len(weekly_values)
    daily_min = min(daily_values)
    daily_max = max(daily_values)

    logger.info(f"Price stats - Daily avg: {daily_avg:.2f}p, Weekly avg: {weekly_avg:.2f}p")
    logger.info(f"Price range - Min: {daily_min:.2f}p, Max: {daily_max:.2f}p")

    return daily_avg, weekly_avg, daily_min, daily_max


@dataclass
class PricePeriod:
    """Represents a contiguous period of prices crossing a threshold."""
    start: datetime
    end: datetime
    avg_price: float
    is_high: bool  # True for high price, False for low price


def find_threshold_periods(
    prices: List[PricePoint],
    weekly_avg: float,
    high_threshold: Optional[NotifyThreshold],
    low_threshold: Optional[NotifyThreshold]
) -> List[PricePeriod]:
    """
    Find contiguous periods where price crosses configured thresholds.

    High price periods that fall within the fixed peak window (16:00-19:00 local time)
    are excluded since users already know about this predictable peak.

    Args:
        prices: List of price points (sorted chronologically)
        weekly_avg: Weekly average price for percentage calculations
        high_threshold: Threshold for high price notifications
        low_threshold: Threshold for low price notifications

    Returns:
        List of PricePeriod objects for threshold crossings
    """
    # Fixed peak pricing window (16:00-19:00 local time)
    PEAK_START_HOUR = 16
    PEAK_END_HOUR = 19

    if not high_threshold and not low_threshold:
        return []

    periods: List[PricePeriod] = []

    def get_price_datetime(price: PricePoint) -> datetime:
        if isinstance(price.valid_from, str):
            dt = datetime.fromisoformat(price.valid_from.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        # valid_from must be datetime if not str
        assert isinstance(price.valid_from, datetime)
        if price.valid_from.tzinfo is None:
            return price.valid_from.replace(tzinfo=timezone.utc)
        return price.valid_from.astimezone(timezone.utc)

    def get_price_end_datetime(price: PricePoint) -> datetime:
        if isinstance(price.valid_to, str):
            dt = datetime.fromisoformat(price.valid_to.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        # valid_to must be datetime if not str
        assert isinstance(price.valid_to, datetime)
        if price.valid_to.tzinfo is None:
            return price.valid_to.replace(tzinfo=timezone.utc)
        return price.valid_to.astimezone(timezone.utc)

    def is_within_peak_window(start: datetime, end: datetime) -> bool:
        """Check if a period falls entirely within the peak pricing window (16:00-19:00)."""
        # Convert to local time
        start_local = start.astimezone()
        end_local = end.astimezone()

        # Check if the period is entirely within the peak window on the same day
        start_hour = start_local.hour
        end_hour = end_local.hour
        end_minute = end_local.minute

        # Period must start at or after 16:00 and end at or before 19:00
        # on the same calendar day
        if start_local.date() != end_local.date():
            return False

        starts_in_peak = start_hour >= PEAK_START_HOUR
        # End hour 19 with 0 minutes means exactly 19:00, which is the boundary
        ends_in_peak = end_hour < PEAK_END_HOUR or (end_hour == PEAK_END_HOUR and end_minute == 0)

        return starts_in_peak and ends_in_peak

    def is_above_high(price_value: float) -> bool:
        if not high_threshold:
            return False
        if high_threshold.threshold_abs is not None:
            return price_value > high_threshold.threshold_abs
        if high_threshold.threshold_pct is not None:
            return price_value > weekly_avg * high_threshold.threshold_pct
        return False

    def is_below_low(price_value: float) -> bool:
        if not low_threshold:
            return False
        if low_threshold.threshold_abs is not None:
            return price_value < low_threshold.threshold_abs
        if low_threshold.threshold_pct is not None:
            return price_value < weekly_avg * low_threshold.threshold_pct
        return False

    # Track current period being built
    current_period_start: Optional[datetime] = None
    current_period_prices: List[float] = []
    current_is_high: Optional[bool] = None

    for i, price in enumerate(prices):
        price_value = price.value_inc_vat
        above_high = is_above_high(price_value)
        below_low = is_below_low(price_value)

        # Determine current state
        if above_high:
            state_is_high: Optional[bool] = True
        elif below_low:
            state_is_high = False
        else:
            state_is_high = None

        # If state changed, close current period and start new one
        if state_is_high != current_is_high:
            # Close current period if one was open
            if current_period_start is not None and current_is_high is not None:
                # End time is the start of the current price point
                period_end = get_price_datetime(price)
                avg_price = sum(current_period_prices) / len(current_period_prices)
                period = PricePeriod(
                    start=current_period_start,
                    end=period_end,
                    avg_price=avg_price,
                    is_high=current_is_high
                )
                # Skip high price periods within the fixed peak window (16:00-19:00)
                if period.is_high and is_within_peak_window(period.start, period.end):
                    logger.debug(
                        f"Skipping high price period {period.start} - {period.end} "
                        "within fixed peak window"
                    )
                else:
                    periods.append(period)

            # Start new period if we're in a threshold state
            if state_is_high is not None:
                current_period_start = get_price_datetime(price)
                current_period_prices = [price_value]
                current_is_high = state_is_high
            else:
                current_period_start = None
                current_period_prices = []
                current_is_high = None
        elif state_is_high is not None:
            # Continue current period
            current_period_prices.append(price_value)

    # Close final period if still open
    if current_period_start is not None and current_is_high is not None and current_period_prices:
        period_end = get_price_end_datetime(prices[-1])
        avg_price = sum(current_period_prices) / len(current_period_prices)
        period = PricePeriod(
            start=current_period_start,
            end=period_end,
            avg_price=avg_price,
            is_high=current_is_high
        )
        # Skip high price periods within the fixed peak window (16:00-19:00)
        if period.is_high and is_within_peak_window(period.start, period.end):
            logger.debug(
                f"Skipping high price period {period.start} - {period.end} "
                "within fixed peak window"
            )
        else:
            periods.append(period)

    logger.debug(f"Found {len(periods)} price threshold periods for notifications")
    return periods


def schedule_price_notifications(
    periods: List[PricePeriod],
    ntfy_client: NtfyClient,
    high_threshold: Optional[NotifyThreshold],
    low_threshold: Optional[NotifyThreshold],
    quiet_window: Optional[TimeRange] = None
) -> None:
    """
    Schedule notifications for price threshold periods.

    Uses ntfy's delay feature to schedule notifications for the start of each period.

    Args:
        periods: List of price periods crossing thresholds
        ntfy_client: Configured ntfy client
        high_threshold: High price threshold (for message context)
        low_threshold: Low price threshold (for message context)
        quiet_window: Optional time range during which notifications should not be sent
    """
    now = datetime.now(timezone.utc)

    for period in periods:
        # Skip periods that have already started
        if period.start <= now:
            logger.debug(f"Skipping past period starting at {period.start}")
            continue

        # Skip periods that start during the quiet window
        if quiet_window:
            period_start_local = period.start.astimezone()
            period_start_time = dt_time(period_start_local.hour, period_start_local.minute)
            if quiet_window.contains(period_start_time):
                logger.debug(
                    f"Skipping notification at {period_start_local.strftime('%H:%M')} "
                    "during quiet window"
                )
                continue

        # Calculate duration in minutes
        duration_mins = int((period.end - period.start).total_seconds() / 60)
        duration_str = f"{duration_mins // 60}h {duration_mins % 60}m" if duration_mins >= 60 else f"{duration_mins}m"

        if period.is_high:
            emoji = "📈"
            threshold_str = str(high_threshold) if high_threshold else "high"
            title = f"{emoji} High Price Alert"
            message = (
                f"Electricity price is above {threshold_str}\n"
                f"**Price:** {period.avg_price:.2f}p/kWh\n"
                f"**Duration:** {duration_str}\n"
                f"**Until:** {period.end.astimezone().strftime('%H:%M')}"
            )
            tags = ["chart_with_upwards_trend", "warning"]
        else:
            emoji = "📉"
            threshold_str = str(low_threshold) if low_threshold else "low"
            title = f"{emoji} Low Price Alert"
            message = (
                f"Electricity price is below {threshold_str}\n"
                f"**Price:** {period.avg_price:.2f}p/kWh\n"
                f"**Duration:** {duration_str}\n"
                f"**Until:** {period.end.astimezone().strftime('%H:%M')}"
            )
            tags = ["chart_with_downwards_trend", "moneybag"]

        try:
            success = ntfy_client.send(
                message=message,
                title=title,
                tags=tags,
                markdown=True,
                delay=period.start
            )
            if success:
                logger.info(
                    f"Scheduled {'high' if period.is_high else 'low'} price notification "
                    f"for {period.start.astimezone().strftime('%H:%M')}"
                )
            else:
                logger.warning(f"Failed to schedule notification for {period.start}")
        except Exception as e:
            logger.error(f"Error scheduling notification: {e}")


def determine_target_temperature(
    price: PricePoint,
    daily_avg: float,
    weekly_avg: float,
    tiers: List[TemperatureTier],
    default_temp: float,
    eco_threshold_pct: Optional[float] = None,
    eco_threshold_abs: Optional[float] = None
) -> Tuple[Optional[float], bool]:
    """
    Determine target temperature or ECO mode for a given price point.

    Args:
        price: Price point to evaluate
        daily_avg: Average price for the day
        weekly_avg: Average price for the week
        tiers: List of temperature tiers (should be sorted by threshold, highest first)
        default_temp: Default temperature if no tier matches
        eco_threshold_pct: Percentage threshold for ECO mode
        eco_threshold_abs: Absolute price threshold for ECO mode

    Returns:
        Tuple of (temperature, eco_mode) where temperature is None if ECO mode is enabled
    """
    price_value = price.value_inc_vat

    # Check if price exceeds ECO threshold
    if eco_threshold_abs is not None:
        eco_threshold = eco_threshold_abs
    elif eco_threshold_pct is not None:
        eco_threshold = max(daily_avg, weekly_avg) * eco_threshold_pct
    else:
        # Default to 133%
        eco_threshold = max(daily_avg, weekly_avg) * 1.33

    if price_value > eco_threshold:
        return (None, True)  # ECO mode

    # Check each tier (tiers should be sorted highest temperature first)
    for tier in tiers:
        # Calculate threshold for this tier
        if tier.threshold_abs is not None:
            threshold = tier.threshold_abs
        elif tier.threshold_pct is not None:
            # Use minimum of averages for lower price thresholds
            threshold = min(daily_avg, weekly_avg) * tier.threshold_pct
        else:
            continue

        # If price is below this tier's threshold, use this tier's temperature
        if price_value < threshold:
            return (tier.temperature, False)

    # No tier matched, use default temperature
    return (default_temp, False)


def calculate_heating_schedule(
    prices: List[PricePoint],
    weekly_prices: List[PricePoint],
    config: Config,
    start_time: datetime
) -> List[HeatingAction]:
    """
    Calculate optimal heating schedule for the next 24 hours.

    Strategy:
    - Uses temperature tiers to set different temperatures based on price levels
    - ECO mode enabled when price exceeds threshold
    - Smooth transitions between temperature levels

    Args:
        prices: Price points for next 24 hours
        weekly_prices: Price points for preceding week
        config: Application configuration
        start_time: When the schedule starts (typically 10pm)

    Returns:
        List of HeatingAction objects in chronological order
    """
    actions = []

    # Calculate statistics
    daily_avg, weekly_avg, daily_min, daily_max = calculate_price_statistics(
        prices, weekly_prices
    )

    # Sort tiers by temperature (highest first) for proper tier matching
    sorted_tiers = sorted(config.temperature_tiers, key=lambda t: t.temperature, reverse=True)

    # Helper function to get datetime from price (always UTC)
    def get_price_datetime(price: PricePoint) -> datetime:
        if isinstance(price.valid_from, str):
            dt = datetime.fromisoformat(price.valid_from.replace('Z', '+00:00'))
            # Ensure it's UTC
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        # valid_from must be datetime if not str
        assert isinstance(price.valid_from, datetime)
        if price.valid_from.tzinfo is None:
            return price.valid_from.replace(tzinfo=timezone.utc)
        return price.valid_from.astimezone(timezone.utc)

    # Determine target for each price point
    price_targets = [
        (p, *determine_target_temperature(
            p, daily_avg, weekly_avg, sorted_tiers, config.default_temp,
            config.eco_threshold_pct, config.eco_threshold_abs
        ))
        for p in prices
    ]

    # Group consecutive periods with same target
    periods: List[PeriodDict] = []
    if price_targets:
        current_temp, current_eco = price_targets[0][1], price_targets[0][2]
        current_start = get_price_datetime(price_targets[0][0])

        for i, (price, temp, eco_mode) in enumerate(price_targets):
            if temp != current_temp or eco_mode != current_eco:
                # Target changed - save the previous period
                periods.append({
                    'temperature': current_temp,
                    'eco_mode': current_eco,
                    'start': current_start,
                    'end': get_price_datetime(price),
                })
                current_temp, current_eco = temp, eco_mode
                current_start = get_price_datetime(price)

            # Save the last period
            if i == len(price_targets) - 1:
                periods.append({
                    'temperature': temp,
                    'eco_mode': eco_mode,
                    'start': current_start,
                    'end': get_price_datetime(price) + timedelta(minutes=30),
                })

    # Generate actions based on periods
    previous_temp: Optional[float] = None
    previous_eco: Optional[bool] = None
    temp_before_eco: Optional[float] = None  # Track temperature before ECO started

    for period in periods:
        period_temp = period['temperature']
        period_eco = period['eco_mode']
        period_start = period['start']

        # Skip if no change from previous period
        if period_temp == previous_temp and period_eco == previous_eco:
            continue

        # Generate appropriate action
        if period_eco:
            # Enable ECO mode
            if not previous_eco:
                # Remember what temperature was set before ECO
                temp_before_eco = previous_temp
                actions.append(HeatingAction(
                    timestamp=period_start,
                    temperature=None,
                    eco_mode=True,
                    reason=f"HIGH price period ({period_start.strftime('%H:%M')})"
                ))
        else:
            # Disable ECO mode if it was on
            if previous_eco:
                actions.append(HeatingAction(
                    timestamp=period_start,
                    temperature=None,
                    eco_mode=False,
                    reason=f"End of HIGH price period - disabling ECO mode"
                ))
                # Only set temperature if it's different from what it was before ECO
                if period_temp is not None and period_temp != temp_before_eco:
                    actions.append(HeatingAction(
                        timestamp=period_start,
                        temperature=period_temp,
                        eco_mode=False,
                        reason=f"Temperature tier: {period_temp}°C ({period_start.strftime('%H:%M')})"
                    ))
                temp_before_eco = None
            elif period_temp != previous_temp and period_temp is not None:
                # Normal temperature change (not related to ECO mode)
                actions.append(HeatingAction(
                    timestamp=period_start,
                    temperature=period_temp,
                    eco_mode=False,
                    reason=f"Temperature tier: {period_temp}°C ({period_start.strftime('%H:%M')})"
                ))

        previous_temp = period_temp
        previous_eco = period_eco

    # Sort actions by timestamp
    actions.sort(key=lambda a: a.timestamp)

    # Filter out actions in quiet window if configured
    if config.quiet_window:
        filtered_actions = []

        for action in actions:
            # Convert to local time for comparison with quiet window
            local_time = action.timestamp
            if local_time.tzinfo is not None:
                local_time = local_time.astimezone()

            # Check if action falls within quiet window
            action_time = dt_time(local_time.hour, local_time.minute)
            in_quiet_window = config.quiet_window.contains(action_time)

            # Only filter temperature-setting actions during quiet window
            # ECO mode changes are allowed to prevent wasting energy
            if in_quiet_window and action.temperature is not None:
                logger.info(f"Skipping temperature change in quiet window: {action}")
            else:
                filtered_actions.append(action)

        actions = filtered_actions

    # Log the schedule
    logger.debug(f"Generated heating schedule with {len(actions)} actions")
    for action in actions:
        logger.info(f"  {action}")

    return actions


def execute_heating_action(
    action: HeatingAction,
    client: NestThermostatClient
) -> None:
    """
    Execute a heating action on the thermostat.

    Only executes if the thermostat is in HEAT or HEATCOOL mode.
    If the thermostat is in OFF, COOL, or any other mode, the action is skipped.

    Args:
        action: The heating action to execute
        client: Nest thermostat client (device already selected)
    """
    try:
        # Check current thermostat mode
        status = client.get_device()
        current_mode = status.mode

        # Only proceed if thermostat is in HEAT or HEATCOOL mode
        if current_mode not in [ThermostatMode.HEAT.value, ThermostatMode.HEATCOOL.value]:
            logger.warning(
                f"Skipping action - thermostat is in {current_mode} mode "
                f"(expected HEAT or HEATCOOL). Action: {action.reason}"
            )
            return

        if action.eco_mode:
            if status.eco_mode != EcoMode.MANUAL_ECO.value:
                logger.info(f"Enabling ECO mode: {action.reason}")
                notify(b"STATUS=Enabling ECO mode")
                client.set_eco_mode(EcoMode.MANUAL_ECO)
        else:
            # First disable ECO mode if it's on
            if status.eco_mode != EcoMode.OFF.value:
                logger.info(f"Disabling ECO mode")
                notify(b"STATUS=Disabling ECO mode")
                client.set_eco_mode(EcoMode.OFF)

            # Only set temperature if specified (temperature=None means just turn off ECO)
            if action.temperature is not None:
                logger.info(f"Setting temperature to {action.temperature}°C: {action.reason}")
                # Set the temperature
                client.set_heat(action.temperature)
                notify(f"STATUS=Set temperature to {action.temperature}°C".encode())
            else:
                logger.info(f"ECO mode disabled: {action.reason}")
    except Exception as e:
        logger.error(f"Failed to execute heating action: {e}")
        raise


def find_cheapest_windows(
    prices: List[PricePoint],
    window_hours: int = 2,
    num_windows: int = 2,
    min_gap_hours: int = 10,
    active_period: Optional[TimeRange] = None
) -> List[Tuple[datetime, datetime, float]]:
    """
    Find the cheapest time windows for running a device.

    Args:
        prices: List of price points (must be sorted chronologically)
        window_hours: Duration of each window in hours
        num_windows: Number of windows to find
        min_gap_hours: Minimum gap between windows in hours
        active_period: Optional TimeRange to restrict windows

    Returns:
        List of tuples (start_time, end_time, avg_price) sorted by start time
    """
    if len(prices) < window_hours * 2:  # Need at least 2 price points per hour (30-min intervals)
        logger.warning(f"Not enough price data to find {window_hours}-hour windows")
        return []

    # Calculate average price for each possible window
    slots_per_window = window_hours * 2  # 2 slots per hour (30-min intervals)
    windows = []

    for i in range(len(prices) - slots_per_window + 1):
        window_prices = prices[i:i + slots_per_window]

        # Get start and end times (ensure UTC)
        if isinstance(window_prices[0].valid_from, str):
            start_time = datetime.fromisoformat(window_prices[0].valid_from.replace('Z', '+00:00'))
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            else:
                start_time = start_time.astimezone(timezone.utc)
        else:
            start_time = window_prices[0].valid_from
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            else:
                start_time = start_time.astimezone(timezone.utc)

        if isinstance(window_prices[-1].valid_to, str):
            end_time = datetime.fromisoformat(window_prices[-1].valid_to.replace('Z', '+00:00'))
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            else:
                end_time = end_time.astimezone(timezone.utc)
        else:
            end_time = window_prices[-1].valid_to
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            else:
                end_time = end_time.astimezone(timezone.utc)

        # Calculate average price for this window
        avg_price = sum(p.value_inc_vat for p in window_prices) / len(window_prices)

        # Check if window is within active period (if configured)
        if active_period is not None:
            # Convert window times to local time for comparison (uses system local timezone)
            window_start_local = start_time.astimezone()
            window_end_local = end_time.astimezone()

            # Check if BOTH window start AND end are within active period
            start_time_only = dt_time(window_start_local.hour, window_start_local.minute)
            end_time_only = dt_time(window_end_local.hour, window_end_local.minute)

            # Use contains_window which has inclusive end semantics
            in_active_period = active_period.contains_window(start_time_only, end_time_only)

            if not in_active_period:
                continue  # Skip this window

        windows.append((start_time, end_time, avg_price, i))

    # Sort by price (cheapest first)
    windows.sort(key=lambda x: x[2])

    # Select windows with minimum gap constraint
    selected: List[Tuple[datetime, datetime, float, int]] = []
    min_gap_slots = min_gap_hours * 2  # Convert hours to 30-min slots

    for start_time, end_time, avg_price, slot_index in windows:
        # Check if this window conflicts with already selected windows
        conflicts = False
        for selected_start, selected_end, _, selected_index in selected:
            # Check if windows are too close together
            gap = abs(slot_index - selected_index)
            if gap < min_gap_slots:
                conflicts = True
                break

        if not conflicts:
            selected.append((start_time, end_time, avg_price, slot_index))

            if len(selected) >= num_windows:
                break

    # Sort by start time for easier reading
    selected.sort(key=lambda x: x[0])

    # Return without the slot_index
    result = [(start, end, price) for start, end, price, _ in selected]

    logger.info(f"Found {len(result)} optimal {window_hours}-hour windows:")
    for i, (start, end, price) in enumerate(result, 1):
        logger.info(f"  Window {i}: {start.strftime('%H:%M')}-{end.strftime('%H:%M')} @ {price:.2f}p/kWh avg")

    return result


def _existing_program_has_pending_slots(
    tg_client: SupplyMasterClient,
    program_id: str,
    now_local: datetime
) -> bool:
    """
    Check if an existing program has any slots that haven't happened yet today.

    Args:
        tg_client: Authenticated TG SupplyMaster client
        program_id: ID of the program to check
        now_local: Current local time

    Returns:
        True if any enabled slot has an end time that hasn't passed yet today
    """
    try:
        program = tg_client.get_program(program_id)

        # Get today's day of week (0=Sunday in TG format)
        # Python: Monday=0, Sunday=6. TG: Sunday=0, Saturday=6
        python_weekday = now_local.weekday()  # Monday=0
        tg_weekday = (python_weekday + 1) % 7  # Convert to TG format (Sunday=0)
        today = DayOfWeek(tg_weekday)

        current_time = now_local.strftime("%H:%M")

        for slot in program.slots:
            # Skip disabled slots
            if not slot.start.enable:
                continue

            # Check if this slot is active today
            if not slot.days.get(today, False):
                continue

            # If the slot's end time hasn't passed yet, the program has pending slots
            # Compare times as strings (HH:MM format works for comparison)
            if slot.end.time > current_time:
                logger.debug(
                    f"Found pending slot: {slot.start.time}-{slot.end.time} "
                    f"(current time: {current_time})"
                )
                return True

        return False

    except Exception as e:
        logger.warning(f"Failed to check existing program: {e}")
        # If we can't check, err on the side of updating
        return False


def program_tg_switch(
    config: Config,
    windows: List[Tuple[datetime, datetime, float]],
    program_name: str = "Agile Optimized"
) -> None:
    """
    Program TG SupplyMaster switch with optimal time windows.

    Args:
        config: Application configuration
        windows: List of (start_time, end_time, avg_price) tuples
        program_name: Name for the program
    """
    if not config.tg_username or not config.tg_password:
        logger.debug("TG SupplyMaster not configured, skipping")
        return

    if not windows:
        logger.warning("No windows to program for TG switch")
        return

    try:
        logger.debug(f"Programming TG SupplyMaster switch with {len(windows)} windows")

        with SupplyMasterClient(
            username=config.tg_username,
            password=config.tg_password
        ) as tg_client:

            # Set device_id if device_name is configured
            if config.tg_device_name:
                devices = tg_client.list_devices()
                for device in devices:
                    if device.name == config.tg_device_name:
                        tg_client.device_id = device.device_id
                        logger.debug(f"Selected TG device: {device.name} ({device.device_id})")
                        break
                else:
                    logger.warning(f"TG device '{config.tg_device_name}' not found")
                    return

            # All days enabled
            all_days = {day: True for day in DayOfWeek}

            # Create program slots from windows
            slots = []
            for start_time, end_time, _ in windows:
                slot = ProgramSlot(
                    start=TimeSlot(enable=True, time=start_time.strftime("%H:%M")),
                    end=TimeSlot(enable=True, time=end_time.strftime("%H:%M")),
                    days=all_days
                )
                slots.append(slot)

            # Fill remaining slots (must have 6 total)
            empty_days = {day: False for day in DayOfWeek}
            while len(slots) < 6:
                slots.append(ProgramSlot(
                    start=TimeSlot(enable=False, time="00:00"),
                    end=TimeSlot(enable=False, time="00:00"),
                    days=empty_days
                ))

            # Get existing programs to find one with matching name, or find an unused slot
            programs_list = tg_client.list_programs()
            program_id = None

            # First, check if a program with this name already exists
            for prog in programs_list.get('namelist', []):
                if prog.get('name') == program_name:
                    program_id = prog['id']
                    logger.debug(f"Found existing program '{program_name}' (ID: {program_id})")
                    break

            # If an existing program was found, check if it has pending slots
            if program_id is not None:
                now_local = datetime.now().astimezone()
                if _existing_program_has_pending_slots(tg_client, program_id, now_local):
                    logger.info(
                        f"Existing TG program '{program_name}' has pending slots today, "
                        f"skipping update to avoid disruption"
                    )
                    return
                logger.debug(f"Existing program '{program_name}' has no pending slots, will update")

            # If not found, find an unused program slot (one with empty name)
            if program_id is None:
                for prog in programs_list.get('namelist', []):
                    if not prog.get('name'):  # Empty name means unused
                        program_id = prog['id']
                        logger.debug(f"Using unused program slot {program_id} for '{program_name}'")
                        break
                else:
                    # No unused slot found, default to program 1
                    program_id = "1"
                    logger.debug(f"No unused slot found, using program {program_id} for '{program_name}'")

            # Create and update program
            program = Program(
                id=program_id,
                name=program_name,
                slots=slots
            )

            tg_client.update_program(program)
            logger.info(f"Updated TG program '{program_name}' with {len(windows)} time windows")

            # Enable the program
            tg_client.enable_program(program_id)
            logger.info(f"Enabled TG program '{program_name}'")

    except Exception as e:
        logger.error(f"Failed to program TG switch: {e}", exc_info=True)
        # Don't raise - TG switch is optional, don't fail the whole cycle


async def run_daily_cycle(
    config: Config,
    nest: NestThermostatClient
) -> list[asyncio.TimerHandle]:
    """
    Run one daily optimization cycle.

    1. Fetch next 24 hours of prices
    2. Fetch previous week of prices for comparison
    3. Calculate optimal heating schedule
    4. Program TG SupplyMaster switch (if configured)
    5. Schedule all actions using call_later()

    Args:
        config: Application configuration
        nest: Nest thermostat client (device already selected)

    Returns:
        List of TimerHandle objects for scheduled actions (can be canceled)
    """
    logger.debug("=" * 60)
    logger.debug("Starting daily heating optimization cycle")
    logger.debug("=" * 60)

    handles: list[asyncio.TimerHandle] = []

    # Initialize Octopus client
    with OctopusEnergyClient(
        api_key=config.api_key,
        account_number=config.account_number,
        mpan=config.mpan
    ) as octopus:
        # Device auto-selected during client initialization
        logger.debug(f"Using thermostat: {nest.device_id}")

        # Fetch prices for next 24 hours (use UTC for all calculations)
        now = datetime.now(timezone.utc)
        tomorrow = now + timedelta(days=1)
        logger.debug(f"Fetching prices from {now.isoformat()} to {tomorrow.isoformat()}")

        daily_prices = octopus.get_unit_rates(
            tariff_code=config.tariff_code,
            period_from=now.isoformat(),
            period_to=tomorrow.isoformat()
        )

        if not daily_prices:
            logger.error("No prices available for next 24 hours")
            return handles

        logger.debug(f"Fetched {len(daily_prices)} price points for next 24 hours")

        # Fetch previous week of prices for comparison
        week_ago = now - timedelta(days=7)
        logger.debug(f"Fetching historical prices from {week_ago.isoformat()}")

        weekly_prices = octopus.get_unit_rates(
            tariff_code=config.tariff_code,
            period_from=week_ago.isoformat(),
            period_to=now.isoformat()
        )

        logger.debug(f"Fetched {len(weekly_prices)} price points for previous week")

        # Schedule price notifications if ntfy is configured
        if config.ntfy_topic and (config.ntfy_high_threshold or config.ntfy_low_threshold):
            try:
                # Calculate weekly average for percentage thresholds
                weekly_avg = sum(p.value_inc_vat for p in weekly_prices) / len(weekly_prices)

                # Find periods crossing thresholds
                threshold_periods = find_threshold_periods(
                    daily_prices,
                    weekly_avg,
                    config.ntfy_high_threshold,
                    config.ntfy_low_threshold
                )

                if threshold_periods:
                    ntfy_client = NtfyClient(
                        topic=config.ntfy_topic,
                        server=config.ntfy_server,
                        token=config.ntfy_token
                    )
                    schedule_price_notifications(
                        threshold_periods,
                        ntfy_client,
                        config.ntfy_high_threshold,
                        config.ntfy_low_threshold,
                        config.quiet_window
                    )
                    logger.info(f"Scheduled {len(threshold_periods)} price notifications")
            except Exception as e:
                logger.error(f"Failed to schedule price notifications: {e}", exc_info=True)
                # Continue with other operations even if notifications fail

        # Program TG SupplyMaster switch if configured
        if config.tg_username and config.tg_password:
            try:
                cheap_windows = find_cheapest_windows(
                    daily_prices,
                    window_hours=config.tg_window_hours,
                    num_windows=config.tg_num_windows,
                    min_gap_hours=config.tg_min_gap_hours,
                    active_period=config.tg_active_period
                )
                if cheap_windows:
                    program_tg_switch(config, cheap_windows)
            except Exception as e:
                logger.error(f"Failed to program TG switch: {e}", exc_info=True)
                # Continue with thermostat scheduling even if TG fails

        # Calculate heating schedule
        actions = calculate_heating_schedule(
            daily_prices,
            weekly_prices,
            config,
            now
        )

        if not actions:
            logger.warning("No heating actions scheduled")
            return handles

        # Schedule all actions using call_later (all times in UTC)
        loop = asyncio.get_running_loop()
        current_time = datetime.now(timezone.utc)

        logger.info(f"Scheduling {len(actions)} heating actions")
        for action in actions:
            # Ensure action timestamp is UTC
            action_time = action.timestamp
            if action_time.tzinfo is None:
                action_time = action_time.replace(tzinfo=timezone.utc)
            else:
                action_time = action_time.astimezone(timezone.utc)

            # Calculate delay until this action
            delay_seconds = (action_time - current_time).total_seconds()

            if delay_seconds < 0:
                logger.warning(f"Skipping past action: {action}")
                continue

            logger.debug(f"Scheduling action in {delay_seconds/60:.1f} minutes: {action}")

            # Schedule the action
            handle = loop.call_later(
                delay_seconds,
                execute_heating_action,
                action,
                nest
            )
            handles.append(handle)

        logger.info(f"Scheduled {len(handles)} heating actions")
        return handles


def setup_signal_handlers(shutdown_event: asyncio.Event, reload_event: asyncio.Event) -> None:
    """Setup signal handlers that set corresponding events."""
    loop = asyncio.get_running_loop()

    def handle_shutdown() -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.debug("Received shutdown signal, initiating graceful shutdown")
        shutdown_event.set()

    def handle_reload() -> None:
        """Handle SIGHUP to reload configuration."""
        logger.debug("Received SIGHUP, will reload configuration")
        reload_event.set()

    # Register signal handlers
    loop.add_signal_handler(signal.SIGINT, handle_shutdown)
    loop.add_signal_handler(signal.SIGTERM, handle_shutdown)
    loop.add_signal_handler(signal.SIGHUP, handle_reload)


def print_price_graph(prices: List[PricePoint]) -> None:
    """
    Print a text-based graph of price fluctuations by half hour.

    Args:
        prices: List of price points to graph
    """
    if not prices:
        return

    # Extract values for graphing
    values = [p.value_inc_vat for p in prices]
    min_price = min(values)
    max_price = max(values)
    price_range = max_price - min_price

    # Graph dimensions
    graph_height = 15
    graph_width = min(len(prices), 48)  # Show up to 48 half-hour periods (24 hours)

    # Sample prices if we have more than graph_width points
    if len(prices) > graph_width:
        step = len(prices) / graph_width
        sampled_prices = [prices[int(i * step)] for i in range(graph_width)]
    else:
        sampled_prices = prices[:graph_width]

    print("\n" + "=" * 70)
    print("  PRICE FLUCTUATION (next 24 hours)")
    print("=" * 70)
    print()

    # Create the graph
    for row in range(graph_height, -1, -1):
        # Calculate the price value this row represents
        if price_range > 0:
            row_price = min_price + (price_range * row / graph_height)
        else:
            row_price = min_price

        # Check if this row has any data points
        has_data = False
        for price in sampled_prices:
            if price_range > 0:
                normalized_value = (price.value_inc_vat - min_price) / price_range * graph_height
            else:
                normalized_value = graph_height / 2
            if abs(normalized_value - row) < 0.5:
                has_data = True
                break

        # Y-axis label (show price on rows with data points)
        if has_data or row == graph_height or row == 0:
            print(f"  {row_price:6.1f}p |", end="")
        else:
            print(f"          |", end="")

        # Plot the data points
        for price in sampled_prices:
            if price_range > 0:
                normalized_value = (price.value_inc_vat - min_price) / price_range * graph_height
            else:
                normalized_value = graph_height / 2

            if abs(normalized_value - row) < 0.5:
                print("*", end="")
            else:
                print(" ", end="")

        print()

    # X-axis
    print("          +" + "-" * len(sampled_prices))

    # Time labels (convert UTC to local time for display)
    print("           ", end="")
    for i, price in enumerate(sampled_prices):
        if i % 8 == 0:  # Show label every 4 hours
            if isinstance(price.valid_from, str):
                time = datetime.fromisoformat(price.valid_from.replace('Z', '+00:00'))
            else:
                time = price.valid_from
            # Convert to local time for display
            if time.tzinfo is not None:
                time = time.astimezone()
            hour = time.strftime("%H:%M")
            print(hour, end=" " * (8 - len(hour) + 1))
    print("\n")


def run_dry_run(config: Config) -> int:
    """
    Dry run mode: Fetch prices, calculate schedule, and display without executing.

    Args:
        config: Application configuration

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("\n" + "=" * 70)
    print("  HEATING OPTIMIZATION - DRY RUN MODE")
    print("=" * 70)
    print()

    # Initialize Octopus client
    with OctopusEnergyClient(
        api_key=config.api_key,
        account_number=config.account_number,
        mpan=config.mpan
    ) as octopus:
        try:
            # Fetch prices for next 24 hours (use UTC internally)
            now = datetime.now(timezone.utc)
            tomorrow = now + timedelta(days=1)
            # Display in local time
            now_local = now.astimezone()
            tomorrow_local = tomorrow.astimezone()
            print(f"📊 Fetching prices from {now_local.strftime('%Y-%m-%d %H:%M')} to {tomorrow_local.strftime('%Y-%m-%d %H:%M')}...")

            daily_prices = octopus.get_unit_rates(
                tariff_code=config.tariff_code,
                period_from=now.isoformat(),
                period_to=tomorrow.isoformat()
            )

            if not daily_prices:
                print("\n❌ ERROR: No prices available for next 24 hours")
                return 1

            print(f"   ✓ Retrieved {len(daily_prices)} price points for next 24 hours")

            # Fetch previous week of prices for comparison
            week_ago = now - timedelta(days=7)
            print(f"\n📊 Fetching historical prices from {week_ago.strftime('%Y-%m-%d')} for comparison...")

            weekly_prices = octopus.get_unit_rates(
                tariff_code=config.tariff_code,
                period_from=week_ago.isoformat(),
                period_to=now.isoformat()
            )

            print(f"   ✓ Retrieved {len(weekly_prices)} price points for previous week")

            # Calculate price statistics
            daily_avg, weekly_avg, daily_min, daily_max = calculate_price_statistics(
                daily_prices, weekly_prices
            )

            print("\n" + "=" * 70)
            print("  PRICE ANALYSIS")
            print("=" * 70)
            print(f"\n  Tariff Code:     {config.tariff_code}")
            print(f"  Daily Average:   {daily_avg:.2f}p/kWh")
            print(f"  Weekly Average:  {weekly_avg:.2f}p/kWh")
            print(f"  Daily Minimum:   {daily_min:.2f}p/kWh")
            print(f"  Daily Maximum:   {daily_max:.2f}p/kWh")
            print(f"  Temperature Tiers: {len(config.temperature_tiers)} tier(s)")
            for i, tier in enumerate(config.temperature_tiers, 1):
                threshold_str = f"{tier.threshold_pct*100:.0f}%" if tier.threshold_pct else f"{tier.threshold_abs}p"
                print(f"    Tier {i}: {tier.temperature}°C @ {threshold_str}")
            print(f"  Default Temp:    {config.default_temp}°C")
            if config.quiet_window:
                print(f"  Quiet Window:    {config.quiet_window}")

            # Print price graph
            print_price_graph(daily_prices)

            # Calculate heating schedule
            actions = calculate_heating_schedule(
                daily_prices,
                weekly_prices,
                config,
                now
            )

            if not actions:
                print("\n⚠️  No heating actions scheduled")
                return 0

            print("\n" + "=" * 70)
            print(f"  PLANNED SCHEDULE ({len(actions)} actions)")
            print("=" * 70)
            print()

            # Helper to find price at a given timestamp (all comparisons in UTC)
            def find_price_at(timestamp: datetime, prices: List[PricePoint]) -> Optional[PricePoint]:
                """Find the price point active at a given timestamp."""
                # Ensure timestamp is UTC
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = timestamp.astimezone(timezone.utc)

                for price in prices:
                    if isinstance(price.valid_from, str):
                        valid_from = datetime.fromisoformat(price.valid_from.replace('Z', '+00:00'))
                        valid_to = datetime.fromisoformat(price.valid_to.replace('Z', '+00:00'))
                    else:
                        valid_from = price.valid_from
                        valid_to = price.valid_to

                    # Ensure UTC for comparison
                    if valid_from.tzinfo is None:
                        valid_from = valid_from.replace(tzinfo=timezone.utc)
                    else:
                        valid_from = valid_from.astimezone(timezone.utc)

                    if valid_to.tzinfo is None:
                        valid_to = valid_to.replace(tzinfo=timezone.utc)
                    else:
                        valid_to = valid_to.astimezone(timezone.utc)

                    if valid_from <= timestamp < valid_to:
                        return price
                return None

            # Pretty print the schedule (convert to local time for display)
            for i, action in enumerate(actions, 1):
                # Convert to local time for display
                local_time = action.timestamp
                if local_time.tzinfo is not None:
                    local_time = local_time.astimezone()
                time_str = local_time.strftime("%a %H:%M")

                # Find the price at this action time
                price = find_price_at(action.timestamp, daily_prices)
                price_str = f" @ {price.value_inc_vat:.2f}p/kWh" if price else ""

                if action.eco_mode:
                    mode_str = "🟦 ECO MODE"
                    temp_str = ""
                else:
                    mode_str = "🔥 HEATING"
                    temp_str = f" → {action.temperature}°C"

                print(f"  {i:2d}. {time_str}  {mode_str}{temp_str}{price_str}")
                print(f"      └─ {action.reason}")
                print()

            print("=" * 70)
            print(f"\n✓ Dry run complete. {len(actions)} actions would be executed.")
            print("  (Run without --dry-run to execute the schedule)\n")

            # Calculate and display TG SupplyMaster schedule
            print("=" * 70)
            print("  TG SUPPLYMASTER PROGRAM")
            print("=" * 70)
            print()

            try:
                cheap_windows = find_cheapest_windows(
                    daily_prices,
                    window_hours=config.tg_window_hours,
                    num_windows=config.tg_num_windows,
                    min_gap_hours=config.tg_min_gap_hours,
                    active_period=config.tg_active_period
                )

                if cheap_windows:
                    print(f"  Device:  {config.tg_device_name or 'Auto-selected'}")
                    print(f"  Program: Agile Optimized")
                    print()

                    for i, (start, end, avg_price) in enumerate(cheap_windows, 1):
                        time_range = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                        print(f"  Slot {i}: {time_range} @ {avg_price:.2f}p/kWh avg")
                        print(f"         (All days)")
                        print()

                    print()
                else:
                    print("  ⚠️  Could not find optimal windows (insufficient price data)")
                    print()

            except Exception as e:
                print(f"  ❌ Error calculating TG schedule: {e}")
                print()

            # Display ntfy notifications that would be sent
            if config.ntfy_topic and (config.ntfy_high_threshold or config.ntfy_low_threshold):
                print("=" * 70)
                print("  NTFY PRICE NOTIFICATIONS")
                print("=" * 70)
                print()
                print(f"  Topic:          {config.ntfy_topic}")
                print(f"  Server:         {config.ntfy_server}")
                if config.ntfy_high_threshold:
                    print(f"  High Threshold: {config.ntfy_high_threshold}")
                if config.ntfy_low_threshold:
                    print(f"  Low Threshold:  {config.ntfy_low_threshold}")
                print()

                try:
                    threshold_periods = find_threshold_periods(
                        daily_prices,
                        weekly_avg,
                        config.ntfy_high_threshold,
                        config.ntfy_low_threshold
                    )

                    if threshold_periods:
                        for i, period in enumerate(threshold_periods, 1):
                            start_local = period.start.astimezone()
                            end_local = period.end.astimezone()
                            duration_mins = int((period.end - period.start).total_seconds() / 60)
                            duration_str = f"{duration_mins // 60}h {duration_mins % 60}m" if duration_mins >= 60 else f"{duration_mins}m"

                            if period.is_high:
                                emoji = "📈"
                                label = "HIGH"
                                threshold = config.ntfy_high_threshold
                            else:
                                emoji = "📉"
                                label = "LOW"
                                threshold = config.ntfy_low_threshold

                            print(f"  {i}. {emoji} {label} price alert @ {start_local.strftime('%a %H:%M')}")
                            print(f"     Price: {period.avg_price:.2f}p/kWh (threshold: {threshold})")
                            print(f"     Duration: {duration_str} (until {end_local.strftime('%H:%M')})")
                            print()
                    else:
                        print("  ℹ️  No price periods crossing thresholds found")
                        print()

                except Exception as e:
                    print(f"  ❌ Error finding notification periods: {e}")
                    print()

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            logger.error(f"Dry run failed: {e}", exc_info=True)
            return 1

    return 0


async def async_main() -> int:
    """
    Main daemon loop (async).

    Runs continuously, executing optimization cycle at 10pm each day.
    Handles signals:
    - SIGINT/SIGTERM: Graceful shutdown
    - SIGHUP: Reload configuration
    """
    # Create events for signal handling
    shutdown_event = asyncio.Event()
    reload_event = asyncio.Event()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Heating Optimization Daemon - optimizes heating schedule based on electricity prices'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: searches /etc, /run, /usr/lib)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Fetch prices and show planned schedule without executing actions'
    )
    parser.add_argument(
        '--tariff-code',
        type=str,
        default=None,
        help='Octopus Energy tariff code (overrides config file)'
    )
    parser.add_argument(
        '--price-threshold',
        type=parse_temperature_tier,
        action='append',
        dest='price_threshold',
        default=None,
        help='Price threshold in format <temp>@<price> (can be specified multiple times). '
             'Examples: --price-threshold "22@50%%" --price-threshold "20@75%%" --price-threshold "18@15p"'
    )
    parser.add_argument(
        '--default-temp',
        type=float,
        default=None,
        help='Default temperature when no tier matches (default: 17.0°C)'
    )
    parser.add_argument(
        '--eco-threshold',
        type=str,
        default=None,
        help='Price threshold for ECO mode with suffix: use %% for percentage (e.g., "133%%") or p for absolute pence (e.g., "25p"). Default: 133%%'
    )
    parser.add_argument(
        '--tg-username',
        type=str,
        default=None,
        help='TG SupplyMaster username (overrides config file)'
    )
    parser.add_argument(
        '--tg-password',
        type=str,
        default=None,
        help='TG SupplyMaster password (overrides credentials file)'
    )
    parser.add_argument(
        '--tg-device-name',
        type=str,
        default=None,
        help='TG SupplyMaster device name (overrides config file)'
    )
    parser.add_argument(
        '--tg-window-hours',
        type=int,
        default=None,
        help='TG window duration in hours (default: 2)'
    )
    parser.add_argument(
        '--tg-num-windows',
        type=int,
        default=None,
        help='TG number of windows per day (default: 2)'
    )
    parser.add_argument(
        '--tg-min-gap-hours',
        type=int,
        default=None,
        help='TG minimum gap between windows in hours (default: 10)'
    )
    parser.add_argument(
        '--tg-active-period',
        type=parse_tg_active_period,
        default=None,
        help='TG active period time range in format hh:mm-hh:mm (e.g., "04:00-20:00"). '
             'TG switch will only be programmed to turn on during this window.'
    )
    parser.add_argument(
        '--quiet-window',
        type=parse_quiet_window,
        default=None,
        help='Quiet window time range in format hh:mm-hh:mm (e.g., "23:00-07:00"). '
             'No heating actions will be scheduled during this window.'
    )
    parser.add_argument(
        '--cycle-time',
        type=parse_cycle_time,
        default=None,
        help='Time of day to run daily optimization cycle in format hh:mm (e.g., "21:50"). '
             'Default: 21:50'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: WARNING)'
    )
    # ntfy notification arguments
    parser.add_argument(
        '--ntfy-topic',
        type=str,
        default=None,
        help='ntfy topic for price notifications'
    )
    parser.add_argument(
        '--ntfy-server',
        type=str,
        default=None,
        help='ntfy server URL (default: https://ntfy.sh)'
    )
    parser.add_argument(
        '--ntfy-token',
        type=str,
        default=None,
        help='ntfy access token for authentication'
    )
    parser.add_argument(
        '--ntfy-high-threshold',
        type=lambda x: parse_notify_threshold(x, "ntfy-high-threshold"),
        default=None,
        help='Send notification when price goes above this threshold. '
             'Use %% suffix for percentage of weekly avg (e.g., "120%%") or p for pence (e.g., "25p")'
    )
    parser.add_argument(
        '--ntfy-low-threshold',
        type=lambda x: parse_notify_threshold(x, "ntfy-low-threshold"),
        default=None,
        help='Send notification when price goes below this threshold. '
             'Use %% suffix for percentage of weekly avg (e.g., "80%%") or p for pence (e.g., "10p")'
    )
    args = parser.parse_args()

    logger.debug("Heating Optimization Daemon starting")

    # Setup signal handlers
    setup_signal_handlers(shutdown_event, reload_event)

    # Load configuration (skip if dry-run with tariff-code)
    config_path = args.config
    config = None

    if args.dry_run and args.tariff_code:
        # Dry-run mode with tariff-code: create minimal config
        logger.debug("Dry-run mode with --tariff-code, skipping config file")

        # Use provided tiers or create default
        if args.price_threshold:
            temperature_tiers = args.price_threshold
        else:
            temperature_tiers = [TemperatureTier(temperature=20.0, threshold_pct=0.75, threshold_abs=None)]

        config = Config(
            tariff_code=args.tariff_code,
            temperature_tiers=temperature_tiers,
            default_temp=args.default_temp if args.default_temp is not None else 17.0,
            thermostat_name="",
            client_id="",
            client_secret="",
            refresh_token="",
            project_id="",
            tg_username=args.tg_username,
            tg_password=args.tg_password,
            tg_device_name=args.tg_device_name,
            tg_window_hours=args.tg_window_hours if args.tg_window_hours is not None else 2,
            tg_num_windows=args.tg_num_windows if args.tg_num_windows is not None else 2,
            tg_min_gap_hours=args.tg_min_gap_hours if args.tg_min_gap_hours is not None else 10,
            tg_active_period=args.tg_active_period,
            quiet_window=args.quiet_window,
            logging_level=args.log_level.upper() if args.log_level is not None else 'WARNING',
            cycle_time=args.cycle_time if args.cycle_time is not None else dt_time(21, 50),
            ntfy_topic=args.ntfy_topic,
            ntfy_server=args.ntfy_server if args.ntfy_server is not None else "https://ntfy.sh",
            ntfy_token=args.ntfy_token,
            ntfy_high_threshold=args.ntfy_high_threshold,
            ntfy_low_threshold=args.ntfy_low_threshold,
        )

        # Apply eco threshold if provided
        if args.eco_threshold:
            eco_str = args.eco_threshold.strip()
            if eco_str.endswith('%'):
                config.eco_threshold_pct = float(eco_str[:-1]) / 100.0
                config.eco_threshold_abs = None
            elif eco_str.endswith('p'):
                config.eco_threshold_abs = float(eco_str[:-1])
                config.eco_threshold_pct = None

        # Configure logging with the specified level
        configure_logging(config.logging_level)
    else:
        # Load full configuration
        try:
            config = load_config(config_path)

            # Override with command line arguments if specified
            apply_cli_overrides(config, args)

            # Configure logging with the specified level
            configure_logging(config.logging_level)
            logger.debug("Configuration loaded successfully")
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            return 1

    # Dry run mode: fetch prices, show schedule, and exit
    if args.dry_run:
        return run_dry_run(config)

    notify_ready()

    # Initialize Nest client once at startup
    logger.debug("Initializing Nest thermostat client")
    nest_client = NestThermostatClient(
        project_id=config.project_id,
        refresh_token=config.refresh_token,
        client_id=config.client_id,
        client_secret=config.client_secret,
        display_name=config.thermostat_name
    )
    logger.debug(f"Nest client initialized, using device: {nest_client.device_id}")

    # Track scheduled actions and next cycle handle
    scheduled_handles: list[asyncio.TimerHandle] = []
    next_cycle_handle: Optional[asyncio.TimerHandle] = None
    loop = asyncio.get_running_loop()

    def schedule_next_cycle() -> None:
        """Schedule the next daily cycle at configured time in local time."""
        nonlocal next_cycle_handle

        # Ensure config is not None
        assert config is not None, "Config must be loaded before scheduling cycle"

        # Calculate next cycle time in local time, then convert to UTC for calculation
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(timezone.utc)

        # Find next cycle time in local time
        next_run_local = now_local.replace(
            hour=config.cycle_time.hour,
            minute=config.cycle_time.minute,
            second=0,
            microsecond=0
        )

        # If it's already past the cycle time, schedule for tomorrow
        if now_local >= next_run_local:
            next_run_local += timedelta(days=1)

        # Convert to UTC for the actual delay calculation
        next_run_utc = next_run_local.astimezone(timezone.utc)

        delay_seconds = (next_run_utc - now_utc).total_seconds()
        logger.debug(f"Next cycle scheduled for {next_run_local.strftime('%Y-%m-%d %H:%M %Z')} (in {delay_seconds/3600:.1f} hours)")

        next_cycle_handle = loop.call_later(delay_seconds, run_cycle_callback)

    def run_cycle_callback() -> None:
        """Callback to run the daily cycle and schedule the next one."""
        nonlocal scheduled_handles, next_cycle_handle

        # Ensure config is not None
        assert config is not None, "Config must be loaded before running cycle"

        try:
            # Cancel any previously scheduled actions before starting new cycle
            if scheduled_handles:
                logger.debug(f"Canceling {len(scheduled_handles)} previous scheduled actions")
                for handle in scheduled_handles:
                    handle.cancel()
                scheduled_handles.clear()

            # Run the daily cycle and get scheduled actions
            # We need to run this in the event loop since it's async
            task = asyncio.create_task(run_daily_cycle(config, nest_client))

            def on_cycle_complete(future: asyncio.Future[list[asyncio.TimerHandle]]) -> None:
                nonlocal scheduled_handles
                try:
                    handles = future.result()
                    scheduled_handles = handles
                    # Schedule the next cycle
                    schedule_next_cycle()
                except Exception as e:
                    logger.error(f"Error in daily cycle: {e}", exc_info=True)
                    # Retry in 1 hour
                    logger.debug("Scheduling retry in 1 hour")
                    next_cycle_handle = loop.call_later(3600, run_cycle_callback)

            task.add_done_callback(on_cycle_complete)

        except Exception as e:
            logger.error(f"Error starting daily cycle: {e}", exc_info=True)
            # Retry in 1 hour
            logger.debug("Scheduling retry in 1 hour")
            next_cycle_handle = loop.call_later(3600, run_cycle_callback)

    # Run the first cycle immediately on startup
    logger.debug("Running initial daily cycle")
    run_cycle_callback()

    # Main event loop - wait for signals using asyncio
    try:
        while True:
            # Wait for either shutdown or reload signal
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            reload_task = asyncio.create_task(reload_event.wait())

            done, pending = await asyncio.wait(
                [shutdown_task, reload_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Handle shutdown
            if shutdown_event.is_set():
                break

            # Handle config reload
            if reload_event.is_set():
                reload_event.clear()

                # Cancel any pending scheduled actions
                if scheduled_handles:
                    logger.debug(f"Canceling {len(scheduled_handles)} scheduled actions due to reload")
                    for handle in scheduled_handles:
                        handle.cancel()
                    scheduled_handles.clear()

                # Cancel the next cycle schedule
                if next_cycle_handle:
                    next_cycle_handle.cancel()

                notify_reloading()
                logger.debug("Reloading configuration")
                try:
                    config = load_config(config_path)
                    logger.debug("Configuration reloaded successfully")

                    # Override with command line arguments if specified
                    apply_cli_overrides(config, args)

                    # Close the old Nest client and create a new one with updated config
                    logger.debug("Closing old Nest client")
                    nest_client.close()

                    logger.debug("Reinitializing Nest thermostat client with new configuration")
                    nest_client = NestThermostatClient(
                        project_id=config.project_id,
                        refresh_token=config.refresh_token,
                        client_id=config.client_id,
                        client_secret=config.client_secret,
                        display_name=config.thermostat_name
                    )

                    logger.debug(f"Nest client reinitialized, using device: {nest_client.device_id}")
                except ConfigurationError as e:
                    logger.error(f"Failed to reload configuration: {e}")
                    logger.debug("Continuing with previous configuration")

                notify_ready()

                # Restart the cycle with new config
                run_cycle_callback()
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)

    # Cancel the next cycle schedule
    if next_cycle_handle:
        next_cycle_handle.cancel()

    # Cancel any remaining scheduled actions
    if scheduled_handles:
        logger.debug(f"Canceling {len(scheduled_handles)} remaining scheduled actions")
        for handle in scheduled_handles:
            handle.cancel()
        scheduled_handles.clear()

    # Close the nest client
    nest_client.close()

    notify_stopping()

    logger.debug("Daemon shutdown complete")
    return 0


def main() -> int:
    """
    Synchronous wrapper for async_main.
    """
    return asyncio.run(async_main())


if __name__ == "__main__":
    exit(main())
