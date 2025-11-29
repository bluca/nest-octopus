#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""
Example usage of the Octopus Energy API client.

This script demonstrates how to use the OctopusEnergyClient to fetch
electricity pricing data and perform basic analysis.
"""

from nest_octopus.octopus import OctopusEnergyClient, OctopusAPIError


def main():
    """Example usage of the Octopus Energy API client."""

    # Configuration parameters
    PRODUCT_CODE = 'AGILE-24-10-01'
    TARIFF_CODE = 'E-1R-AGILE-24-10-01-N'
    PERIOD_FROM = '2025-11-29T00:00Z'
    PERIOD_TO = '2025-11-30T00:00Z'

    # Create client and fetch data
    with OctopusEnergyClient() as client:
        try:
            print(f"Fetching electricity rates for {PRODUCT_CODE}")
            print(f"Period: {PERIOD_FROM} to {PERIOD_TO}\n")

            # Get price points
            price_points = client.get_unit_rates(
                product_code=PRODUCT_CODE,
                tariff_code=TARIFF_CODE,
                period_from=PERIOD_FROM,
                period_to=PERIOD_TO
            )

            print(f"Retrieved {len(price_points)} price points:\n")

            # Display prices
            for point in price_points[:5]:  # Show first 5
                print(f"  {point.valid_from} to {point.valid_to}")
                print(f"    Price (inc VAT): {point.value_inc_vat:.4f}p/kWh")
                print(f"    Price (exc VAT): {point.value_exc_vat:.4f}p/kWh\n")

            if len(price_points) > 5:
                print(f"  ... and {len(price_points) - 5} more\n")

            # Calculate statistics
            prices_inc_vat = [p.value_inc_vat for p in price_points]

            if prices_inc_vat:
                avg_price = sum(prices_inc_vat) / len(prices_inc_vat)
                min_price = min(prices_inc_vat)
                max_price = max(prices_inc_vat)

                print("Price Statistics:")
                print(f"  Average: {avg_price:.4f}p/kWh")
                print(f"  Minimum: {min_price:.4f}p/kWh")
                print(f"  Maximum: {max_price:.4f}p/kWh")
                print(f"  Range:   {max_price - min_price:.4f}p/kWh")

                # Find cheapest period
                cheapest_idx = prices_inc_vat.index(min_price)
                cheapest_point = price_points[cheapest_idx]
                print(f"\nCheapest period: {cheapest_point.valid_from} to {cheapest_point.valid_to}")

        except OctopusAPIError as e:
            print(f"Error fetching data: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
