# SPDX-License-Identifier: MPL-2.0
"""
Heating Optimization Daemon

Queries electricity prices daily at 8pm and optimizes heating schedule to minimize costs
while maintaining comfort. Uses dynamic pricing to determine when to heat (22°C),
maintain comfort (17°C), or enable ECO mode during high prices.

The daemon:
1. Fetches electricity prices for the next 24 hours at 8pm
2. Analyzes prices against daily and weekly averages
3. Calculates optimal heating schedule
4. Executes temperature/mode changes at calculated times
5. Sleeps between changes until next action
6. Repeats cycle daily at 8pm
"""

import argparse
import configparser
import errno
import logging
import os
import signal
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from nest_octopus.nest_thermostat import EcoMode, NestThermostatClient, ThermostatMode
from nest_octopus.octopus import OctopusEnergyClient, PricePoint

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

# Global flags for signal handling
shutdown_requested = False
reload_config_requested = False


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


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
    low_price_temp: float = 20.0
    average_price_temp: float = 17.0


@dataclass
class HeatingAction:
    """Represents a scheduled heating change."""
    timestamp: datetime
    temperature: Optional[float]
    eco_mode: bool
    reason: str

    def __repr__(self):
        if self.eco_mode:
            return f"HeatingAction({self.timestamp.strftime('%H:%M')}, ECO, {self.reason})"
        return f"HeatingAction({self.timestamp.strftime('%H:%M')}, {self.temperature}°C, {self.reason})"


class PriceCategory:
    """Price classification categories."""
    LOW = "LOW"
    AVERAGE = "AVERAGE"
    HIGH = "HIGH"


def notify(message):
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


def notify_ready():
    notify(b"READY=1")


def notify_reloading():
    microsecs = time.clock_gettime_ns(time.CLOCK_MONOTONIC) // 1000
    notify(f"RELOADING=1\nMONOTONIC_USEC={microsecs}".encode())


def notify_stopping():
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

    parser = configparser.ConfigParser()
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
        )

        # Optional heating preferences
        if parser.has_option('heating', 'low_price_temp'):
            config.low_price_temp = parser.getfloat('heating', 'low_price_temp')
        if parser.has_option('heating', 'average_price_temp'):
            config.average_price_temp = parser.getfloat('heating', 'average_price_temp')

        # Validate Octopus Energy configuration
        if not config.tariff_code and not (config.api_key and config.account_number):
            raise ConfigurationError(
                "Either tariff_code or both api_key and account_number must be configured"
            )

        return config

    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise ConfigurationError(f"Invalid configuration: {e}")


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


def classify_price(
    price: PricePoint,
    daily_avg: float,
    weekly_avg: float
) -> str:
    """
    Classify a price point as LOW, AVERAGE, or HIGH.

    Args:
        price: Price point to classify
        daily_avg: Average price for the day
        weekly_avg: Average price for the week

    Returns:
        Price category (LOW, AVERAGE, or HIGH)
    """
    # Calculate threshold for LOW prices
    # LOW: below both daily and weekly averages
    avg_threshold = min(daily_avg, weekly_avg) * 0.75

    # Calculate threshold for HIGH prices
    # HIGH: significantly above both averages
    high_threshold = max(daily_avg, weekly_avg) * 1.33

    if price.value_inc_vat < avg_threshold:
        return PriceCategory.LOW
    elif price.value_inc_vat > high_threshold:
        return PriceCategory.HIGH
    else:
        return PriceCategory.AVERAGE


def calculate_heating_schedule(
    prices: List[PricePoint],
    weekly_prices: List[PricePoint],
    config: Config,
    start_time: datetime
) -> List[HeatingAction]:
    """
    Calculate optimal heating schedule for the next 24 hours.

    Strategy:
    - HIGH prices: Enable ECO mode to minimize costs
    - LOW price periods: Heat to 22°C for comfort when cheap
    - AVERAGE prices: Maintain 17°C for basic comfort

    Args:
        prices: Price points for next 24 hours
        weekly_prices: Price points for preceding week
        config: Application configuration
        start_time: When the schedule starts (typically 8pm)

    Returns:
        List of HeatingAction objects in chronological order
    """
    actions = []

    # Calculate statistics
    daily_avg, weekly_avg, daily_min, daily_max = calculate_price_statistics(
        prices, weekly_prices
    )

    # Helper function to get datetime from price
    def get_price_datetime(price):
        if isinstance(price.valid_from, str):
            return datetime.fromisoformat(price.valid_from.replace('Z', '+00:00'))
        return price.valid_from

    # Classify each price point (prices are already sorted chronologically)
    classified = [
        (p, classify_price(p, daily_avg, weekly_avg))
        for p in prices
    ]

    # Group consecutive periods of same category
    periods = []
    current_category = classified[0][1]
    current_start = get_price_datetime(classified[0][0])

    for i, (price, category) in enumerate(classified):
        if category != current_category:
            # Category changed - save the previous period
            periods.append({
                'category': current_category,
                'start': current_start,
                'end': get_price_datetime(price),
                'prices': [p for p, c in classified if c == current_category]
            })
            current_category = category
            current_start = get_price_datetime(price)

        # Save the last period
        if i == len(classified) - 1:
            periods.append({
                'category': category,
                'start': current_start,
                'end': get_price_datetime(price) + timedelta(minutes=30),  # Add duration of last slot
                'prices': [p for p, c in classified if c == category]
            })

    # Generate actions based on periods
    current_mode = None
    previous_category = None

    for i, period in enumerate(periods):
        category = period['category']
        period_start = period['start']

        # If exiting LOW price period, set temperature back to average
        if previous_category == PriceCategory.LOW and category != PriceCategory.LOW:
            if current_mode == 'LOW':
                actions.append(HeatingAction(
                    timestamp=period_start,
                    temperature=config.average_price_temp,
                    eco_mode=False,
                    reason=f"End of LOW price period - returning to {config.average_price_temp}°C"
                ))
                current_mode = 'AVERAGE'

        # If exiting HIGH period, always disable ECO mode first (without setting temperature)
        if previous_category == PriceCategory.HIGH and current_mode == 'ECO':
            actions.append(HeatingAction(
                timestamp=period_start,
                temperature=None,
                eco_mode=False,
                reason=f"End of HIGH price period - disabling ECO mode"
            ))
            current_mode = None

        if category == PriceCategory.LOW:
            # Heat to comfort temperature during low prices
            if current_mode != 'LOW':
                actions.append(HeatingAction(
                    timestamp=period_start,
                    temperature=config.low_price_temp,
                    eco_mode=False,
                    reason=f"LOW price period ({period_start.strftime('%H:%M')})"
                ))
                current_mode = 'LOW'

        elif category == PriceCategory.AVERAGE:
            # Maintain basic comfort during average prices
            if current_mode != 'AVERAGE':
                actions.append(HeatingAction(
                    timestamp=period_start,
                    temperature=config.average_price_temp,
                    eco_mode=False,
                    reason=f"AVERAGE price period ({period_start.strftime('%H:%M')})"
                ))
                current_mode = 'AVERAGE'

        elif category == PriceCategory.HIGH:
            # During HIGH prices, enable ECO mode to minimize costs
            if current_mode != 'ECO':
                actions.append(HeatingAction(
                    timestamp=period_start,
                    temperature=None,
                    eco_mode=True,
                    reason=f"HIGH price period ({period_start.strftime('%H:%M')})"
                ))
                current_mode = 'ECO'

        previous_category = category

    # Sort actions by timestamp
    actions.sort(key=lambda a: a.timestamp)

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
            logger.info(f"Enabling ECO mode: {action.reason}")
            notify(b"STATUS=Enabling ECO mode")
            client.set_eco_mode(EcoMode.MANUAL_ECO)
        else:
            # First disable ECO mode if it's on
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


def run_daily_cycle(config: Config) -> None:
    """
    Run one daily optimization cycle.

    1. Fetch next 24 hours of prices
    2. Fetch previous week of prices for comparison
    3. Calculate optimal heating schedule
    4. Execute scheduled actions with sleep intervals

    Args:
        config: Application configuration
    """
    logger.debug("=" * 60)
    logger.debug("Starting daily heating optimization cycle")
    logger.debug("=" * 60)

    # Initialize clients
    octopus = OctopusEnergyClient(
        api_key=config.api_key,
        account_number=config.account_number,
        mpan=config.mpan
    )
    nest = NestThermostatClient(
        project_id=config.project_id,
        refresh_token=config.refresh_token,
        client_id=config.client_id,
        client_secret=config.client_secret,
        display_name=config.thermostat_name  # Used for device selection
    )

    try:
        # Device auto-selected during client initialization
        logger.debug(f"Using thermostat: {nest.device_id}")

        # Fetch prices for next 24 hours
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        logger.debug(f"Fetching prices from {now.isoformat()} to {tomorrow.isoformat()}")

        daily_prices = octopus.get_unit_rates(
            tariff_code=config.tariff_code,
            period_from=now.isoformat(),
            period_to=tomorrow.isoformat()
        )

        if not daily_prices:
            logger.error("No prices available for next 24 hours")
            return

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

        # Calculate heating schedule
        actions = calculate_heating_schedule(
            daily_prices,
            weekly_prices,
            config,
            now
        )

        if not actions:
            logger.warning("No heating actions scheduled")
            return

        # Execute actions with sleep intervals
        for i, action in enumerate(actions):
            # Calculate sleep time until this action
            now = datetime.now(timezone.utc) if action.timestamp.tzinfo else datetime.now()
            sleep_seconds = (action.timestamp - now).total_seconds()

            if sleep_seconds > 0:
                logger.debug(f"Sleeping for {sleep_seconds/60:.1f} minutes until next action")
                time.sleep(sleep_seconds)

            # Execute the action
            execute_heating_action(action, nest)

            # Check if this is the last action
            if i == len(actions) - 1:
                logger.debug("All scheduled actions completed")

    finally:
        octopus.close()
        nest.close()


def handle_shutdown_signal(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    logger.debug(f"Received {signal_name}, initiating graceful shutdown")
    shutdown_requested = True


def handle_reload_signal(signum, frame):
    """Handle SIGHUP to reload configuration."""
    global reload_config_requested
    logger.debug("Received SIGHUP, will reload configuration")
    reload_config_requested = True


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

    # Time labels
    print("           ", end="")
    for i, price in enumerate(sampled_prices):
        if i % 8 == 0:  # Show label every 4 hours
            if isinstance(price.valid_from, str):
                time = datetime.fromisoformat(price.valid_from.replace('Z', '+00:00'))
            else:
                time = price.valid_from
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
    octopus = OctopusEnergyClient(
        api_key=config.api_key,
        account_number=config.account_number,
        mpan=config.mpan
    )

    try:
        # Fetch prices for next 24 hours
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        print(f"📊 Fetching prices from {now.strftime('%Y-%m-%d %H:%M')} to {tomorrow.strftime('%Y-%m-%d %H:%M')}...")

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
        print(f"  Low Temp:        {config.low_price_temp}°C")
        print(f"  Average Temp:    {config.average_price_temp}°C")

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

        # Helper to find price at a given timestamp
        def find_price_at(timestamp, prices):
            """Find the price point active at a given timestamp."""
            for price in prices:
                if isinstance(price.valid_from, str):
                    valid_from = datetime.fromisoformat(price.valid_from.replace('Z', '+00:00'))
                    valid_to = datetime.fromisoformat(price.valid_to.replace('Z', '+00:00'))
                else:
                    valid_from = price.valid_from
                    valid_to = price.valid_to

                if valid_from <= timestamp < valid_to:
                    return price
            return None

        # Pretty print the schedule
        for i, action in enumerate(actions, 1):
            time_str = action.timestamp.strftime("%a %H:%M")

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

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Dry run failed: {e}", exc_info=True)
        return 1
    finally:
        octopus.close()

    return 0


def main():
    """
    Main daemon loop.

    Runs continuously, executing optimization cycle at 8pm each day.
    Handles signals:
    - SIGINT/SIGTERM: Graceful shutdown
    - SIGHUP: Reload configuration
    """
    global shutdown_requested, reload_config_requested

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
    args = parser.parse_args()

    logger.debug("Heating Optimization Daemon starting")

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGHUP, handle_reload_signal)

    # Load configuration (skip if dry-run with tariff-code)
    config_path = args.config
    config = None

    if args.dry_run and args.tariff_code:
        # Dry-run mode with tariff-code: create minimal config
        logger.debug("Dry-run mode with --tariff-code, skipping config file")
        config = Config(
            tariff_code=args.tariff_code,
            thermostat_name="",
            client_id="",
            client_secret="",
            refresh_token="",
            project_id=""
        )
    else:
        # Load full configuration
        try:
            config = load_config(config_path)
            logger.debug("Configuration loaded successfully")

            # Override tariff code if specified
            if args.tariff_code:
                logger.debug(f"Overriding tariff code with: {args.tariff_code}")
                config.tariff_code = args.tariff_code
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            return 1

    # Dry run mode: fetch prices, show schedule, and exit
    if args.dry_run:
        return run_dry_run(config)

    notify_ready()

    while not shutdown_requested:
        try:
            # Check for config reload request
            if reload_config_requested:
                notify_reloading()
                logger.debug("Reloading configuration")
                try:
                    config = load_config(config_path)
                    logger.debug("Configuration reloaded successfully")
                except ConfigurationError as e:
                    logger.error(f"Failed to reload configuration: {e}")
                    logger.debug("Continuing with previous configuration")
                reload_config_requested = False
                notify_ready()

            # Calculate next 8pm
            now = datetime.now()
            next_run = now.replace(hour=20, minute=0, second=0, microsecond=0)

            # If it's already past 8pm, schedule for tomorrow
            if now.hour >= 20:
                next_run += timedelta(days=1)

            # Sleep until 8pm, checking for signals periodically
            sleep_seconds = (next_run - datetime.now()).total_seconds()
            logger.debug(f"Next cycle at {next_run.isoformat()} (in {sleep_seconds/3600:.1f} hours)")

            # Sleep in small intervals to check for signals
            sleep_interval = min(60, sleep_seconds)  # Check every minute or less
            while sleep_seconds > 0 and not shutdown_requested and not reload_config_requested:
                time.sleep(min(sleep_interval, sleep_seconds))
                sleep_seconds = (next_run - datetime.now()).total_seconds()

            # If shutdown was requested during sleep, exit
            if shutdown_requested:
                break

            # If config reload was requested during sleep, continue to reload
            if reload_config_requested:
                continue

            # Run the daily cycle
            run_daily_cycle(config)

        except Exception as e:
            logger.error(f"Error in daily cycle: {e}", exc_info=True)
            # Sleep for an hour before retrying
            logger.debug("Sleeping for 1 hour before retry")
            sleep_time = 3600
            while sleep_time > 0 and not shutdown_requested:
                time.sleep(min(60, sleep_time))
                sleep_time -= 60

    notify_stopping()

    logger.debug("Daemon shutdown complete")
    return 0


if __name__ == "__main__":
    exit(main())
