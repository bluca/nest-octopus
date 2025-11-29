# SPDX-License-Identifier: MPL-2.0
"""
Octopus Energy API Client Module

This module handles querying electricity tariff data from the Octopus Energy API.
It provides methods to fetch pricing information for specific products and tariffs.
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
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

    def __repr__(self):
        return f"PricePoint(inc_vat={self.value_inc_vat}p, {self.valid_from} to {self.valid_to})"


class OctopusEnergyClient:
    """
    Client for interacting with the Octopus Energy API to query electricity tariff pricing.

    Attributes:
        base_url (str): The base URL of the Octopus Energy API
        timeout (int): Request timeout in seconds
        session (requests.Session): HTTP session for connection pooling
    """

    BASE_URL = "https://api.octopus.energy/v1"

    def __init__(self, timeout: int = 30):
        """
        Initialize the Octopus Energy API client.

        Args:
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = self.BASE_URL
        self.timeout = timeout
        self.session = requests.Session()

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

    def get_unit_rates(
        self,
        tariff_code: str,
        period_from: str,
        period_to: str
    ) -> List[PricePoint]:
        """
        Fetch electricity unit rates for a specific tariff.

        Args:
            tariff_code: Tariff code (e.g., 'E-1R-AGILE-24-10-01-N')
            period_from: Start of period in ISO 8601 format (e.g., '2025-12-01T00:00Z')
            period_to: End of period in ISO 8601 format (e.g., '2025-12-01T04:00Z')

        Returns:
            List of PricePoint objects containing rate information

        Raises:
            OctopusAPIError: If the request fails or returns an error status
            ValueError: If tariff code format is invalid
        """
        # Extract product code from tariff code
        product_code = self.extract_product_code(tariff_code)

        endpoint = f"products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
        params = {
            'period_from': period_from,
            'period_to': period_to
        }

        url = f"{self.base_url}/{endpoint}"

        try:
            logger.info(f"Fetching unit rates from {url} for period {period_from} to {period_to}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Extract results and convert to PricePoint objects
            results = data.get('results', [])
            price_points = [PricePoint(result) for result in results]

            # Sort by valid_from timestamp (API returns in reverse chronological order)
            price_points.sort(key=lambda p: p.valid_from)

            logger.info(f"Successfully retrieved {len(price_points)} price points")
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
        tariff_code: str,
        period_from: str,
        period_to: str
    ) -> Dict[str, Any]:
        """
        Fetch raw JSON response for unit rates.

        Args:
            tariff_code: Tariff code (e.g., 'E-1R-AGILE-24-10-01-N')
            period_from: Start of period in ISO 8601 format
            period_to: End of period in ISO 8601 format

        Returns:
            Raw JSON response as a dictionary

        Raises:
            OctopusAPIError: If the request fails or returns an error status
            ValueError: If tariff code format is invalid
        """
        # Extract product code from tariff code
        product_code = self.extract_product_code(tariff_code)

        endpoint = f"products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
        params = {
            'period_from': period_from,
            'period_to': period_to
        }

        url = f"{self.base_url}/{endpoint}"

        try:
            logger.info(f"Fetching raw data from {url}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Successfully retrieved raw data")
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

    def close(self):
        """Close the HTTP session."""
        self.session.close()
        logger.info("Octopus Energy API client session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
