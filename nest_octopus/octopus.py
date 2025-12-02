# SPDX-License-Identifier: MPL-2.0
"""
Octopus Energy API Client Module

This module handles querying electricity tariff data from the Octopus Energy API.
It provides methods to fetch pricing information for specific products and tariffs.
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OctopusAPIError(Exception):
    """Custom exception for Octopus API client errors."""
    pass


class PricePoint:
    """
    Represents a single price point from the Octopus Energy API.

    Attributes:
        value_exc_vat (float): Price excluding VAT in pence/kWh
        value_inc_vat (float): Price including VAT in pence/kWh
        valid_from (str): ISO 8601 timestamp when this rate starts
        valid_to (str): ISO 8601 timestamp when this rate ends
        payment_method (str|None): Payment method if applicable
    """

    def __init__(self, data: Dict[str, Any]):
        """Initialize from API response data."""
        self.value_exc_vat = data['value_exc_vat']
        self.value_inc_vat = data['value_inc_vat']
        self.valid_from = data['valid_from']
        self.valid_to = data['valid_to']
        self.payment_method = data.get('payment_method')

    def __repr__(self) -> str:
        return f"PricePoint(inc_vat={self.value_inc_vat}p, {self.valid_from} to {self.valid_to})"


class OctopusEnergyClient:
    """
    Client for interacting with the Octopus Energy API to query electricity tariff pricing.

    Attributes:
        base_url (str): The base URL of the Octopus Energy API
        timeout (int): Request timeout in seconds
        session (requests.Session): HTTP session for connection pooling
        api_key (str|None): API key for authenticated requests
        account_number (str|None): Account number for fetching tariff information
        mpan (str|None): Meter Point Administration Number for filtering when multiple meters exist
        _cached_tariff_code (str|None): Cached tariff code
        _cache_expires_at (int|None): Monotonic timestamp when cache expires (nanoseconds)
    """

    BASE_URL = "https://api.octopus.energy/v1"

    def __init__(self, timeout: int = 30, api_key: Optional[str] = None, account_number: Optional[str] = None, mpan: Optional[str] = None):
        """
        Initialize the Octopus Energy API client.

        Args:
            timeout: Request timeout in seconds (default: 30)
            api_key: Optional API key for authenticated requests
            account_number: Optional account number for automatic tariff lookup
            mpan: Optional Meter Point Administration Number for filtering when multiple meters exist
        """
        self.base_url = self.BASE_URL
        self.timeout = timeout
        self.session = requests.Session()
        self.api_key = api_key
        self.account_number = account_number
        self.mpan = mpan
        self._cached_tariff_code: Optional[str] = None
        self._cache_expires_at: Optional[int] = None

    @staticmethod
    def extract_product_code(tariff_code: str) -> str:
        """
        Extract product code from tariff code.

        Tariff codes follow the format: E-1R-PRODUCT-CODE-N
        For example: E-1R-AGILE-24-10-01-N -> AGILE-24-10-01

        Args:
            tariff_code: Full tariff code (e.g., 'E-1R-AGILE-24-10-01-N')

        Returns:
            Product code extracted from tariff (e.g., 'AGILE-24-10-01')

        Raises:
            ValueError: If tariff code format is invalid
        """
        parts = tariff_code.split('-')
        if len(parts) < 4:
            raise ValueError(f"Invalid tariff code format: {tariff_code}")

        # Skip first two parts (e.g., 'E', '1R') and last part (e.g., 'N')
        # Join the middle parts to form product code
        product_code = '-'.join(parts[2:-1])
        return product_code

    def get_current_tariff_code(self) -> str:
        """
        Fetch the current tariff code from the account.

        Uses the account API to retrieve the currently active electricity tariff.
        Caches the result for up to 12 hours or until the tariff expires.

        Returns:
            Current tariff code for the account

        Raises:
            OctopusAPIError: If API key/account number not provided or request fails
        """
        if not self.api_key or not self.account_number:
            raise OctopusAPIError(
                "API key and account number required to fetch tariff code automatically"
            )

        # Check if cache is still valid
        if self._cached_tariff_code and self._cache_expires_at:
            if time.monotonic_ns() < self._cache_expires_at:
                logger.debug(f"Using cached tariff code: {self._cached_tariff_code}")
                return self._cached_tariff_code

        url = f"{self.base_url}/accounts/{self.account_number}/"

        try:
            logger.debug(f"Fetching account information from {url}")
            response = self.session.get(
                url,
                auth=(self.api_key, ''),  # API key as username, empty password
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            # Navigate through the response to find current tariff
            properties = data.get('properties', [])
            if not properties:
                raise OctopusAPIError("No properties found in account")

            # Get the first property (usually only one)
            property_data = properties[0]
            electricity_meter_points = property_data.get('electricity_meter_points', [])

            if not electricity_meter_points:
                raise OctopusAPIError("No electricity meter points found")

            # Filter for non-export (import) meter points
            import_meters = [mp for mp in electricity_meter_points if not mp.get('is_export', False)]

            if not import_meters:
                raise OctopusAPIError("No import electricity meter point found")

            # Handle multiple meters
            meter_point = None
            if len(import_meters) == 1:
                # Only one meter, use it
                meter_point = import_meters[0]
            elif len(import_meters) > 1:
                # Multiple meters - need mpan to filter
                if not self.mpan:
                    mpans = [mp.get('mpan') for mp in import_meters]
                    raise OctopusAPIError(
                        f"Multiple electricity meters found ({len(import_meters)}). "
                        f"Specify 'mpan' parameter to select one. Available MPANs: {', '.join(mpans)}"
                    )
                # Filter by mpan
                for mp in import_meters:
                    if mp.get('mpan') == self.mpan:
                        meter_point = mp
                        break
                if not meter_point:
                    raise OctopusAPIError(
                        f"No meter found with MPAN '{self.mpan}'. "
                        f"Available MPANs: {', '.join(mp.get('mpan') for mp in import_meters)}"
                    )

            assert meter_point is not None
            agreements = meter_point.get('agreements', [])
            if not agreements:
                raise OctopusAPIError("No tariff agreements found")

            # Find the currently valid agreement
            current_agreement = None
            tariff_valid_until = None
            now = datetime.now()

            for agreement in agreements:
                valid_from_str = agreement.get('valid_from')
                valid_to_str = agreement.get('valid_to')

                if not valid_from_str:
                    continue

                # Parse timestamps (convert to naive for comparison with datetime.now())
                valid_from = datetime.fromisoformat(valid_from_str.replace('Z', '+00:00')).replace(tzinfo=None)

                # If valid_to is None, the agreement is open-ended
                if valid_to_str:
                    valid_to = datetime.fromisoformat(valid_to_str.replace('Z', '+00:00')).replace(tzinfo=None)
                    if valid_from <= now < valid_to:
                        current_agreement = agreement
                        tariff_valid_until = valid_to
                        break
                else:
                    # Open-ended agreement, valid if started
                    if valid_from <= now:
                        current_agreement = agreement
                        tariff_valid_until = None  # No expiry
                        break

            if not current_agreement:
                raise OctopusAPIError("No currently valid tariff agreement found")

            tariff_code = current_agreement.get('tariff_code')
            if not tariff_code:
                raise OctopusAPIError("Tariff code not found in agreement")

            logger.debug(f"Found current tariff code: {tariff_code}")

            # Calculate cache expiration time using monotonic clock
            # Cache expires after 12 hours OR when tariff ends, whichever is earlier
            max_cache_duration = 12 * 60 * 60 * 1_000_000_000  # 12 hours in nanoseconds
            cache_expiry_monotonic = time.monotonic_ns() + max_cache_duration

            if tariff_valid_until:
                logger.debug(f"Tariff valid until: {tariff_valid_until.isoformat()}")
                # Calculate nanoseconds until tariff expiry
                time_until_tariff_expiry = int((tariff_valid_until - now).total_seconds() * 1_000_000_000)
                if time_until_tariff_expiry > 0:
                    # Use the earlier of 12 hours or tariff expiry
                    cache_expiry_monotonic = time.monotonic_ns() + min(max_cache_duration, time_until_tariff_expiry)

            self._cached_tariff_code = tariff_code
            self._cache_expires_at = cache_expiry_monotonic
            assert isinstance(tariff_code, str)
            return tariff_code

        except requests.exceptions.Timeout:
            error_msg = f"Request to {url} timed out after {self.timeout} seconds"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

        except (ValueError, KeyError) as e:
            error_msg = f"Invalid JSON response or missing data: {str(e)}"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

    def get_unit_rates(
        self,
        tariff_code: Optional[str] = None,
        period_from: Optional[str] = None,
        period_to: Optional[str] = None
    ) -> List[PricePoint]:
        """
        Fetch electricity unit rates for a specific tariff.

        Args:
            tariff_code: Optional tariff code (e.g., 'E-1R-AGILE-24-10-01-N').
                        If not provided, will be fetched from account using API key.
            period_from: Start of period in ISO 8601 format (e.g., '2025-12-01T00:00Z')
            period_to: End of period in ISO 8601 format (e.g., '2025-12-01T04:00Z')

        Returns:
            List of PricePoint objects containing rate information

        Raises:
            OctopusAPIError: If the request fails or returns an error status
            ValueError: If tariff code format is invalid
        """
        # If tariff_code not provided, fetch from account
        if not tariff_code:
            tariff_code = self.get_current_tariff_code()

        # Extract product code from tariff code
        product_code = self.extract_product_code(tariff_code)

        endpoint = f"products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
        params = {
            'period_from': period_from,
            'period_to': period_to
        }

        url = f"{self.base_url}/{endpoint}"

        try:
            logger.debug(f"Fetching unit rates from {url} for period {period_from} to {period_to}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Extract results and convert to PricePoint objects
            results = data.get('results', [])
            price_points = [PricePoint(result) for result in results]

            # Sort by valid_from timestamp (API returns in reverse chronological order)
            price_points.sort(key=lambda p: p.valid_from)

            logger.debug(f"Successfully retrieved {len(price_points)} price points")
            return price_points

        except requests.exceptions.Timeout:
            error_msg = f"Request to {url} timed out after {self.timeout} seconds"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

        except (ValueError, KeyError) as e:
            error_msg = f"Invalid JSON response: {str(e)}"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

    def get_raw_data(
        self,
        tariff_code: Optional[str] = None,
        period_from: Optional[str] = None,
        period_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch raw JSON response for unit rates.

        Args:
            tariff_code: Optional tariff code (e.g., 'E-1R-AGILE-24-10-01-N').
                        If not provided, will be fetched from account using API key.
            period_from: Start of period in ISO 8601 format
            period_to: End of period in ISO 8601 format

        Returns:
            Raw JSON response as a dictionary

        Raises:
            OctopusAPIError: If the request fails or returns an error status
            ValueError: If tariff code format is invalid
        """
        # If tariff_code not provided, fetch from account
        if not tariff_code:
            tariff_code = self.get_current_tariff_code()

        # Extract product code from tariff code
        product_code = self.extract_product_code(tariff_code)

        endpoint = f"products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
        params = {
            'period_from': period_from,
            'period_to': period_to
        }

        url = f"{self.base_url}/{endpoint}"

        try:
            logger.debug(f"Fetching raw data from {url}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            logger.debug(f"Successfully retrieved raw data")
            assert isinstance(data, dict)
            return data

        except requests.exceptions.Timeout:
            error_msg = f"Request to {url} timed out after {self.timeout} seconds"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

        except ValueError as e:
            error_msg = f"Invalid JSON response: {str(e)}"
            logger.error(error_msg)
            raise OctopusAPIError(error_msg)

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
        logger.debug("Octopus Energy API client session closed")

    def __enter__(self) -> 'OctopusEnergyClient':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
