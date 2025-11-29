"""Test fixture data documentation.

This directory contains JSON fixtures for testing the Octopus Energy API client.

Directory Structure:
-------------------

valid/ - Contains valid API responses for testing success scenarios
  - standard_response.json: Typical API response with 8 price points
  - empty_results.json: Valid response with no results
  - paginated_response.json: Response with pagination (has 'next' URL)
  - high_low_prices.json: Response with extreme price values
  - single_result.json: Response with only one price point
  - negative_prices.json: Response with zero and negative prices (can occur in real data)
  - large_response_46_results.json: Large dataset with 46 price points (original prices.json)

invalid/ - Contains invalid/malformed responses for testing error handling
  - missing_value_exc_vat.json: Missing required field 'value_exc_vat'
  - missing_valid_from.json: Missing required field 'valid_from'
  - missing_results.json: Missing the 'results' array
  - error_response.json: API error response format
  - malformed_json.txt: Completely invalid JSON
  - wrong_type_count.json: 'count' field is string instead of number
  - wrong_type_price.json: Price value is string instead of number
  - array_instead_of_object.json: Root is array instead of object

Usage in Tests:
--------------

Use the load_fixture() helper function to load these files in tests.
All tests should use these fixtures and never make real HTTP requests.
"""
