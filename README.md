# Nest Octopus - Smart Heating Optimization Service

A Python daemon that optimizes heating schedules based on dynamic electricity pricing from Octopus Energy. Automatically controls Google Nest thermostats and TG SupplyMaster devices to minimize energy costs while maintaining comfort.

Boilerplate and unit tests written with Copilot, as life is short. Don't @ me.

## Overview

The heating optimizer automatically:
1. Fetches electricity prices daily at 10pm for the next 24 hours
2. Analyzes prices against daily and weekly averages
3. Calculates optimal heating schedule based on price thresholds
4. Executes temperature and ECO mode changes at calculated times
5. Supports optional quiet windows to prevent temperature changes during sleep hours
6. Integrates with TG SupplyMaster for hot water optimization

## Features

### Smart Scheduling
- **Dynamic price-based optimization**: Heats during cheap periods, reduces heating during expensive periods
- **Multi-threshold classification**: LOW/AVERAGE/HIGH price periods with configurable thresholds
- **Quiet window support**: Prevents temperature changes during sleep hours while still allowing ECO mode
- **Daily automation**: Automatic daily updates at 10pm with systemd integration

### Device Integration
- **Google Nest Thermostat**: Full control via official Google SDM API
- **TG SupplyMaster**: Hot water optimization with configurable heating windows via wi-fi relay switch
- **Dual-mode operation**: Can control either or both systems

## How It Works

### Price Analysis

1. **Fetch prices**: Downloads next 24 hours of electricity prices from Octopus Energy
2. **Calculate statistics**: Computes daily and weekly average prices
3. **Classify periods**: Each price period is classified as:
   - **LOW**: < 75% of average (default: heat to 20°C)
   - **AVERAGE**: 75-133% of average (default: maintain 17°C)
   - **HIGH**: > 133% of average (default: ECO mode)

### Schedule Calculation

4. **Generate actions**: Creates temperature/ECO mode changes at price period boundaries
5. **Apply quiet window**: Filters out temperature changes during quiet hours (ECO mode still allowed)
6. **Optimize TG windows**: Finds cheapest windows for hot water heating if TG configured

### Execution

7. **Execute actions**: Applies each action at its scheduled time
8. **Sleep until next**: Waits for next scheduled change
9. **Daily refresh**: At 10pm, fetches new prices and recalculates schedule

## Project Structure

```
nest-octopus/
├── nest_octopus/
│   ├── __init__.py
│   ├── heating_optimizer.py    # Main daemon and scheduling logic
│   ├── octopus.py              # Octopus Energy API client
│   ├── nest_thermostat.py      # Google Nest SDM API client
│   └── tg_supplymaster.py      # TG SupplyMaster API client
├── tests/                      # unit tests
├── mkosi/                      # build portable service image that can be attached to any systemd system
├── pyproject.toml
└── README.md
```

## Configuration

### Configuration File

Create a configuration file at `/etc/nest-octopus/config.ini` (or `/run/nest-octopus/config.ini` or `/usr/lib/nest-octopus/config.ini`):

```ini
[octopus]
# Optional: can be derived from account
tariff_code = E-1R-AGILE-FLEX-22-11-25-H
# Optional: for authenticated access
account_number = A-12345678
api_key = xxxxxxxxxxxxx
# Optional: in case an account has multiple meters
mpan = 1234567890123

[nest]
# OAUTH2 client id from google
client_id = your-client-id.apps.googleusercontent.com
# SDM project ID from google
project_id = your-project-id
# Optional: in case account has multiple devices
thermostat_name = Living Room

[heating]
low_price_temp = 20.0        # Temperature during cheap periods (°C)
average_price_temp = 17.0    # Temperature during normal periods (°C)
low_price_threshold = 0.75   # Multiplier for low price (75% of average)
high_price_threshold = 1.33  # Multiplier for high price (133% of average)
quiet_window = 23:00-07:00   # No temperature changes during this window

[tg_supplymaster]
# username from TG app
username = your-username
# Optional: in case an account has multiple devices
device_name = your-device
window_hours = 2             # Duration of each heating window
num_windows = 2              # Number of windows per day
min_gap_hours = 10           # Minimum gap between windows

[logging]
level = WARNING              # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Credentials

Credentials are read from `$CREDENTIALS_DIRECTORY` (systemd):

- `client_secret` - Google OAuth client secret (if using Nest)
- `refresh_token` - Google OAuth refresh token (if using Nest)
- `tg_password` - TG SupplyMaster password (if using TG)

## Usage

### Dry-Run Mode

Preview the heating schedule without executing changes:

```bash
# Using config file
python -m nest_octopus.heating_optimizer --dry-run

# Quick test with just tariff code
python -m nest_octopus.heating_optimizer --dry-run --tariff-code E-1R-AGILE-FLEX-22-11-25-H
```

Example output:
```
=== Heating Schedule (Next 24 Hours) ===
20:00 → 22.0°C (LOW price period)
23:00 → 17.0°C (AVERAGE price - quiet window prevents change)
06:00 → ECO (HIGH price period)
08:00 → 17.0°C (AVERAGE price period)
```

### Daemon Mode

Run continuously with automatic daily updates:

```bash
python -m nest_octopus.heating_optimizer
```

### Command Line Options

All configuration parameters can be overridden via CLI:

```bash
# Core options
--config PATH                   # Configuration file path
--dry-run                       # Preview mode without execution
--log-level LEVEL              # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Octopus Energy
--tariff-code CODE             # Override tariff code

# Heating preferences
--low-price-threshold FLOAT    # Low price multiplier (default: 0.75)
--high-price-threshold FLOAT   # High price multiplier (default: 1.33)
--low-price-temp TEMP          # Temperature for cheap periods (default: 20.0)
--average-price-temp TEMP      # Temperature for normal periods (default: 17.0)
--quiet-window HH:MM-HH:MM     # Time range for quiet operation

# TG SupplyMaster
--tg-username USER             # TG username
--tg-password PASS             # TG password
--tg-device-name NAME          # TG device name
--tg-window-hours HOURS        # Window duration (default: 2)
--tg-num-windows N             # Number of windows (default: 2)
--tg-min-gap-hours HOURS       # Minimum gap (default: 10)
```

### systemd Integration

Build portable image with mkosi.

It will use the current distribution/release combination by default,
can be overridden with `--distribution`/`--release`.

```bash
mkosi build
```

An image is also built and published on the (Open Build Service)[https://download.opensuse.org/repositories/home:/bluca:/octpus/debian_13_images/]
and can be automatically installed and kept up to date via systemd-sysupdate.

Once downloaded, attach it to a system, enable it (so it starts on boot) and start it:

```bash
run0 portablectl attach --enable --now /path/to/nest-octopus.raw
```

When changing configuration a reload can be issued:

```bash
run0 systemctl reload nest-octopus.service
```

## Testing

Test and linting coverage via pytest and mypy.

```bash
python3 -m mypy nest_octopus tests
python3 -m pytest
```

## License

Mozilla Public License 2.0 (MPL-2.0)
