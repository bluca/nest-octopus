#SPDX-License-Identifier: MPL-2.0
"""
Unit tests for the Octopus Energy API Client module.

This module contains comprehensive tests for the OctopusEnergyClient class,
using mocking to simulate REST API endpoints without making actual HTTP calls.

ALL TESTS RUN COMPLETELY OFFLINE - No real network requests are ever made.
All test data is loaded from fixture files in tests/fixtures/
"""

import pytest
import json
import os
import socket
from unittest.mock import Mock, patch, MagicMock
from nest_octopus.octopus import OctopusEnergyClient, OctopusAPIError, PricePoint
import requests


# Helper function to load fixture files
def load_fixture(filename):
    """
    Load a JSON fixture file from the fixtures directory.

    Args:
        filename: Name of fixture file (e.g., 'valid/standard_response.json')

    Returns:
        Parsed JSON data or raw text for invalid files
    """
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures', 'octopus')
    filepath = os.path.join(fixtures_dir, filename)

    with open(filepath, 'r') as f:
        if filename.endswith('.txt'):
            return f.read()
        return json.load(f)


@pytest.fixture(autouse=True)
def block_network_access(monkeypatch):
    """
    Pytest fixture that runs for ALL tests and prevents any real network access.

    This fixture monkeypatches socket.socket and socket.getaddrinfo to ensure
    no test can accidentally make a real HTTP request or DNS lookup. If any code
    tries to open a socket or resolve a hostname, it will raise a RuntimeError.
    """
    def guard(*args, **kwargs):
        raise RuntimeError(
            "Network access is blocked in tests! "
            "All HTTP requests must be mocked. "
            "If you see this error, a test is trying to make a real network call."
        )

    monkeypatch.setattr(socket, 'socket', guard)
    monkeypatch.setattr(socket, 'getaddrinfo', guard)


@pytest.fixture
def client():
    """Fixture providing an OctopusEnergyClient instance."""
    return OctopusEnergyClient()


class TestPricePoint:
    """Test cases for the PricePoint class."""

    def test_price_point_initialization(self):
        """Test PricePoint initialization from API data."""
        data = {
            'value_exc_vat': 13.73,
            'value_inc_vat': 14.4165,
            'valid_from': '2025-11-30T22:30:00Z',
            'valid_to': '2025-11-30T23:00:00Z',
            'payment_method': None
        }

        price_point = PricePoint(data)

        assert price_point.value_exc_vat == 13.73
        assert price_point.value_inc_vat == 14.4165
        assert price_point.valid_from == '2025-11-30T22:30:00Z'
        assert price_point.valid_to == '2025-11-30T23:00:00Z'
        assert price_point.payment_method is None

    def test_price_point_repr(self):
        """Test PricePoint string representation."""
        data = {
            'value_exc_vat': 13.73,
            'value_inc_vat': 14.4165,
            'valid_from': '2025-11-30T22:30:00Z',
            'valid_to': '2025-11-30T23:00:00Z',
            'payment_method': None
        }

        price_point = PricePoint(data)
        repr_str = repr(price_point)

        assert '14.4165' in repr_str
        assert '2025-11-30T22:30:00Z' in repr_str


class TestOctopusEnergyClientInitialization:
    """Test cases for OctopusEnergyClient initialization."""

    def test_init_default(self, client):
        """Test default initialization."""
        assert client.base_url == "https://api.octopus.energy/v1"
        assert client.timeout == 30
        assert isinstance(client.session, requests.Session)

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = OctopusEnergyClient(timeout=60)
        assert client.timeout == 60

    def test_base_url_constant(self):
        """Test that base URL is set correctly."""
        assert OctopusEnergyClient.BASE_URL == "https://api.octopus.energy/v1"


class TestExtractProductCode:
    """Test cases for the extract_product_code static method."""

    def test_extract_product_code_valid(self):
        """Test extracting product code from valid tariff codes."""
        # Standard Agile tariff
        assert OctopusEnergyClient.extract_product_code('E-1R-AGILE-24-10-01-N') == 'AGILE-24-10-01'

        # Flex tariff
        assert OctopusEnergyClient.extract_product_code('E-1R-AGILE-FLEX-22-11-25-N') == 'AGILE-FLEX-22-11-25'

        # Go tariff
        assert OctopusEnergyClient.extract_product_code('E-1R-GO-24-04-03-N') == 'GO-24-04-03'

    def test_extract_product_code_invalid_format(self):
        """Test that invalid tariff formats raise ValueError."""
        # Too few parts
        with pytest.raises(ValueError, match="Invalid tariff code format"):
            OctopusEnergyClient.extract_product_code('E-1R-AGILE')

        # Empty string
        with pytest.raises(ValueError, match="Invalid tariff code format"):
            OctopusEnergyClient.extract_product_code('')

        # Only one part
        with pytest.raises(ValueError, match="Invalid tariff code format"):
            OctopusEnergyClient.extract_product_code('INVALID')

    def test_extract_product_code_minimum_parts(self):
        """Test minimum valid tariff code (4 parts)."""
        # Minimum: E-1R-PROD-N
        assert OctopusEnergyClient.extract_product_code('E-1R-PROD-N') == 'PROD'


class TestGetUnitRates:
    """Test cases for the get_unit_rates method."""

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_unit_rates_success(self, mock_get, client):
        """Test successful retrieval of unit rates."""
        # Load fixture data
        fixture_data = load_fixture('valid/large_response_46_results.json')

        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        # Call the method
        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T04:00Z'
        )

        # Verify results
        assert len(price_points) == fixture_data['count']
        assert all(isinstance(p, PricePoint) for p in price_points)

        # Check first price point (now sorted chronologically, so earliest time is first)
        first_point = price_points[0]
        assert first_point.value_exc_vat == 10.18
        assert first_point.value_inc_vat == 10.689
        assert first_point.valid_from == '2025-11-30T00:00:00Z'

        # Check last price point (latest time)
        last_point = price_points[-1]
        assert last_point.value_exc_vat == 13.73
        assert last_point.value_inc_vat == 14.4165

        # Verify the request was made correctly
        expected_url = "https://api.octopus.energy/v1/products/AGILE-24-10-01/electricity-tariffs/E-1R-AGILE-24-10-01-N/standard-unit-rates/"
        mock_get.assert_called_once_with(
            expected_url,
            params={
                'period_from': '2025-12-01T00:00Z',
                'period_to': '2025-12-01T04:00Z'
            },
            timeout=30
        )

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_unit_rates_sorted_chronologically(self, mock_get, client):
        """Test that results are sorted in chronological order."""
        # Create fixture with reverse chronological data (as API returns)
        fixture_data = {
            'count': 3,
            'results': [
                {
                    'value_exc_vat': 15.0,
                    'value_inc_vat': 15.75,
                    'valid_from': '2025-12-01T02:00:00Z',
                    'valid_to': '2025-12-01T02:30:00Z',
                    'payment_method': None
                },
                {
                    'value_exc_vat': 12.0,
                    'value_inc_vat': 12.60,
                    'valid_from': '2025-12-01T01:00:00Z',
                    'valid_to': '2025-12-01T01:30:00Z',
                    'payment_method': None
                },
                {
                    'value_exc_vat': 10.0,
                    'value_inc_vat': 10.50,
                    'valid_from': '2025-12-01T00:00:00Z',
                    'valid_to': '2025-12-01T00:30:00Z',
                    'payment_method': None
                }
            ]
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T03:00Z'
        )

        # Verify prices are sorted chronologically (earliest first)
        assert len(price_points) == 3
        assert price_points[0].valid_from == '2025-12-01T00:00:00Z'
        assert price_points[0].value_inc_vat == 10.50
        assert price_points[1].valid_from == '2025-12-01T01:00:00Z'
        assert price_points[1].value_inc_vat == 12.60
        assert price_points[2].valid_from == '2025-12-01T02:00:00Z'
        assert price_points[2].value_inc_vat == 15.75

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_unit_rates_empty_results(self, mock_get, client):
        """Test handling of empty results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'count': 0,
            'next': None,
            'previous': None,
            'results': []
        }
        mock_get.return_value = mock_response

        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T00:00Z'
        )

        assert len(price_points) == 0
        assert isinstance(price_points, list)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_unit_rates_timeout(self, mock_get, client):
        """Test handling of request timeout."""
        mock_get.side_effect = requests.exceptions.Timeout()

        with pytest.raises(OctopusAPIError) as exc_info:
            client.get_unit_rates(
                tariff_code='E-1R-AGILE-24-10-01-N',
                period_from='2025-12-01T00:00Z',
                period_to='2025-12-01T04:00Z'
            )

        assert 'timed out after 30 seconds' in str(exc_info.value)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_unit_rates_http_404(self, mock_get, client):
        """Test handling of HTTP 404 error (invalid product/tariff code)."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Product not found"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with pytest.raises(OctopusAPIError) as exc_info:
            client.get_unit_rates(
                tariff_code='E-1R-INVALID-N',
                period_from='2025-12-01T00:00Z',
                period_to='2025-12-01T04:00Z'
            )

        assert '404' in str(exc_info.value)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_unit_rates_http_500(self, mock_get, client):
        """Test handling of HTTP 500 error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        with pytest.raises(OctopusAPIError) as exc_info:
            client.get_unit_rates(
                tariff_code='E-1R-AGILE-24-10-01-N',
                period_from='2025-12-01T00:00Z',
                period_to='2025-12-01T04:00Z'
            )

        assert '500' in str(exc_info.value)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_unit_rates_connection_error(self, mock_get, client):
        """Test handling of connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        with pytest.raises(OctopusAPIError) as exc_info:
            client.get_unit_rates(
                tariff_code='E-1R-AGILE-24-10-01-N',
                period_from='2025-12-01T00:00Z',
                period_to='2025-12-01T04:00Z'
            )

        assert 'Request failed' in str(exc_info.value)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_unit_rates_invalid_json(self, mock_get, client):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        with pytest.raises(OctopusAPIError) as exc_info:
            client.get_unit_rates(
                tariff_code='E-1R-AGILE-24-10-01-N',
                period_from='2025-12-01T00:00Z',
                period_to='2025-12-01T04:00Z'
            )

        assert 'Invalid JSON response' in str(exc_info.value)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_unit_rates_different_product_codes(self, mock_get, client):
        """Test with different product and tariff codes."""
        fixture_data = load_fixture('valid/large_response_46_results.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        client.get_unit_rates(
            tariff_code='E-1R-AGILE-FLEX-22-11-25-C',
            period_from='2025-11-30T00:00Z',
            period_to='2025-11-30T23:59Z'
        )

        expected_url = "https://api.octopus.energy/v1/products/AGILE-FLEX-22-11-25/electricity-tariffs/E-1R-AGILE-FLEX-22-11-25-C/standard-unit-rates/"
        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == expected_url


class TestGetRawData:
    """Test cases for the get_raw_data method."""

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_raw_data_success(self, mock_get, client):
        """Test successful retrieval of raw data."""
        fixture_data = load_fixture('valid/large_response_46_results.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        raw_data = client.get_raw_data(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T04:00Z'
        )

        assert raw_data == fixture_data
        assert 'count' in raw_data
        assert 'results' in raw_data
        assert raw_data['count'] == 46

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_get_raw_data_with_pagination(self, mock_get, client):
        """Test raw data includes pagination information."""
        response_with_next = {
            'count': 100,
            'next': 'https://api.octopus.energy/v1/products/.../standard-unit-rates/?page=2',
            'previous': None,
            'results': []
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_with_next
        mock_get.return_value = mock_response

        raw_data = client.get_raw_data(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-31T23:59Z'
        )

        assert raw_data['next'] is not None
        assert 'page=2' in raw_data['next']


class TestContextManager:
    """Test cases for context manager functionality."""

    @patch('nest_octopus.octopus.requests.Session.close')
    def test_context_manager(self, mock_close):
        """Test OctopusEnergyClient can be used as a context manager."""
        with OctopusEnergyClient() as client:
            assert isinstance(client, OctopusEnergyClient)

        # Session should be closed after exiting context
        mock_close.assert_called_once()

    @patch('nest_octopus.octopus.requests.Session.close')
    def test_close_method(self, mock_close):
        """Test explicit close method."""
        client = OctopusEnergyClient()
        client.close()

        mock_close.assert_called_once()


class TestIntegrationScenarios:
    """Integration-style tests with realistic scenarios."""

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_multiple_sequential_requests(self, mock_get, client):
        """Test making multiple sequential requests with the same client."""
        # Setup different responses for different time periods
        response1 = load_fixture('valid/large_response_46_results.json')
        response2 = {
            'count': 1,
            'next': None,
            'previous': None,
            'results': [
                {
                    'value_exc_vat': 20.0,
                    'value_inc_vat': 21.0,
                    'valid_from': '2025-12-02T00:00:00Z',
                    'valid_to': '2025-12-02T00:30:00Z',
                    'payment_method': None
                }
            ]
        }

        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = response1

        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = response2

        mock_get.side_effect = [mock_response1, mock_response2]

        # First request
        prices1 = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T04:00Z'
        )

        # Second request
        prices2 = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-02T00:00Z',
            period_to='2025-12-02T01:00Z'
        )

        assert len(prices1) == 46
        assert len(prices2) == 1
        assert prices2[0].value_inc_vat == 21.0
        assert mock_get.call_count == 2

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_realistic_usage_pattern(self, mock_get, client):
        """Test realistic usage pattern: get rates and process them."""
        fixture_data = load_fixture('valid/large_response_46_results.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        # Get rates
        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T04:00Z'
        )

        # Process them (example calculation)
        total_prices = [p.value_inc_vat for p in price_points]
        avg_price = sum(total_prices) / len(total_prices)
        min_price = min(total_prices)
        max_price = max(total_prices)

        assert avg_price > 0
        assert min_price <= avg_price <= max_price
        assert len(price_points) == 46


class TestValidFixtures:
    """Test cases using valid fixture files."""

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_standard_response_fixture(self, mock_get, client):
        """Test using the standard_response fixture."""
        fixture_data = load_fixture('valid/standard_response.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T04:00Z'
        )

        assert len(price_points) == 8
        # Results are now sorted chronologically (earliest first)
        assert price_points[0].valid_from == '2025-12-01T00:00:00Z'
        assert price_points[0].value_inc_vat == 19.047
        assert price_points[-1].valid_from == '2025-12-01T03:30:00Z'
        assert price_points[-1].value_inc_vat == 14.4165
        assert all(isinstance(p, PricePoint) for p in price_points)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_empty_results_fixture(self, mock_get, client):
        """Test handling of empty results from fixture."""
        fixture_data = load_fixture('valid/empty_results.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T00:00Z'
        )

        assert len(price_points) == 0
        assert isinstance(price_points, list)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_paginated_response_fixture(self, mock_get, client):
        """Test response with pagination."""
        fixture_data = load_fixture('valid/paginated_response.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        raw_data = client.get_raw_data(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-31T23:59Z'
        )

        assert raw_data['count'] == 150
        assert raw_data['next'] is not None
        assert 'page=2' in raw_data['next']

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_high_low_prices_fixture(self, mock_get, client):
        """Test extreme price values."""
        fixture_data = load_fixture('valid/high_low_prices.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T23:59Z'
        )

        prices = [p.value_inc_vat for p in price_points]
        assert min(prices) == 5.775  # Very low price
        assert max(prices) == 54.915  # Very high price

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_single_result_fixture(self, mock_get, client):
        """Test response with single price point."""
        fixture_data = load_fixture('valid/single_result.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T12:00Z',
            period_to='2025-12-01T12:30Z'
        )

        assert len(price_points) == 1
        assert price_points[0].value_inc_vat == 26.25

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_negative_prices_fixture(self, mock_get, client):
        """Test handling of zero and negative prices (can occur in real data)."""
        fixture_data = load_fixture('valid/negative_prices.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T03:00Z',
            period_to='2025-12-01T04:00Z'
        )

        assert len(price_points) == 2
        assert price_points[0].value_inc_vat == 0.0
        assert price_points[1].value_inc_vat == -2.625


class TestLargeResponseFixture:
    """Test cases specifically for the large 46-result dataset."""

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_large_response_46_results(self, mock_get, client):
        """Test handling of large response with 46 price points."""
        fixture_data = load_fixture('valid/large_response_46_results.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-11-30T00:00Z',
            period_to='2025-11-30T23:00Z'
        )

        # Verify we got all 46 results
        assert len(price_points) == 46
        assert fixture_data['count'] == 46

        # Verify first and last price points (now sorted chronologically)
        # First should be earliest time
        assert price_points[0].value_exc_vat == 10.18
        assert price_points[0].value_inc_vat == 10.689
        assert price_points[0].valid_from == '2025-11-30T00:00:00Z'
        # Last should be latest time
        assert price_points[-1].value_exc_vat == 13.73
        assert price_points[-1].value_inc_vat == 14.4165

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_large_response_statistics(self, mock_get, client):
        """Test statistical analysis on large dataset."""
        fixture_data = load_fixture('valid/large_response_46_results.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-11-30T00:00Z',
            period_to='2025-11-30T23:00Z'
        )

        # Calculate statistics
        prices_inc_vat = [p.value_inc_vat for p in price_points]
        avg_price = sum(prices_inc_vat) / len(prices_inc_vat)
        min_price = min(prices_inc_vat)
        max_price = max(prices_inc_vat)

        # Verify reasonable values
        assert len(prices_inc_vat) == 46
        assert min_price < avg_price < max_price
        # Verify price range
        assert min_price > 0  # All prices are positive in this dataset
        assert min_price == 9.639  # Minimum price
        assert max_price == 36.141  # Maximum price (peak pricing)


class TestInvalidFixtures:
    """Test cases using invalid fixture files to ensure proper error handling."""

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_missing_value_exc_vat(self, mock_get, client):
        """Test handling of missing required field."""
        fixture_data = load_fixture('invalid/missing_value_exc_vat.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        with pytest.raises(OctopusAPIError) as exc_info:
            client.get_unit_rates(
                tariff_code='E-1R-AGILE-24-10-01-N',
                period_from='2025-12-01T00:00Z',
                period_to='2025-12-01T04:00Z'
            )

        assert 'Invalid JSON response' in str(exc_info.value)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_missing_valid_from(self, mock_get, client):
        """Test handling of missing timestamp field."""
        fixture_data = load_fixture('invalid/missing_valid_from.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        with pytest.raises(OctopusAPIError) as exc_info:
            client.get_unit_rates(
                tariff_code='E-1R-AGILE-24-10-01-N',
                period_from='2025-12-01T00:00Z',
                period_to='2025-12-01T04:00Z'
            )

        assert 'Invalid JSON response' in str(exc_info.value)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_missing_results_field(self, mock_get, client):
        """Test handling of missing results array - returns empty list gracefully."""
        fixture_data = load_fixture('invalid/missing_results.json')

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = fixture_data
        mock_get.return_value = mock_response

        # The code handles missing results gracefully by returning empty list
        price_points = client.get_unit_rates(
            tariff_code='E-1R-AGILE-24-10-01-N',
            period_from='2025-12-01T00:00Z',
            period_to='2025-12-01T04:00Z'
        )

        # Should return empty list rather than raising error
        assert len(price_points) == 0
        assert isinstance(price_points, list)

    @patch('nest_octopus.octopus.requests.Session.get')
    def test_malformed_json_response(self, mock_get, client):
        """Test handling of completely invalid JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        mock_get.return_value = mock_response

        with pytest.raises(OctopusAPIError) as exc_info:
            client.get_unit_rates(
                tariff_code='E-1R-AGILE-24-10-01-N',
                period_from='2025-12-01T00:00Z',
                period_to='2025-12-01T04:00Z'
            )

        assert 'Invalid JSON response' in str(exc_info.value)


class TestNoNetworkAccess:
    """Test that verifies the network blocking fixture is working."""

    def test_network_access_is_blocked(self):
        """Verify that socket access is blocked by our fixture."""
        import socket

        with pytest.raises(RuntimeError) as exc_info:
            # Try to create a socket - this should fail
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        assert "Network access is blocked in tests" in str(exc_info.value)

    def test_unmocked_request_would_fail(self, client):
        """Verify that an unmocked API call would fail (network is blocked)."""
        # This test verifies that if we forgot to mock, the test would fail
        # rather than making a real network call

        # Don't mock anything - try to make a real call
        with pytest.raises(RuntimeError) as exc_info:
            # This should fail at the socket/DNS level before even trying to connect
            client.get_unit_rates(
                tariff_code='E-1R-AGILE-24-10-01-N',
                period_from='2025-12-01T00:00Z',
                period_to='2025-12-01T04:00Z'
            )

        assert "Network access is blocked in tests" in str(exc_info.value)
