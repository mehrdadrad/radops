import logging
import os

import geoip2.database
from config.tools import tool_settings as settings
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_geoip_location(ip_address: str) -> str:
    """
    Performs a GeoIP lookup for a given IP address to find its geographical location.
    Use this tool when the user asks for the location, country, city, or other geographic details of an IP address.

    Args:
        ip_address: The IP address to look up (e.g., '8.8.8.8').

    Returns:
        A string containing the location details of the IP address, or an error message if the lookup fails.
    """
    if not settings.geoip.database_path:
        return "Error: GeoIP database path is not configured. Please check your tools.yaml."

    db_path = os.path.expanduser(settings.geoip.database_path)
    logger.debug(f"Using GeoIP database at: {db_path}")

    try:
        with geoip2.database.Reader(db_path) as reader:
            response = reader.city(ip_address)
            return (
                f"GeoIP Lookup for {ip_address}:\n"
                f"- Country: {response.country.name} ({response.country.iso_code})\n"
                f"- City: {response.city.name}\n"
                f"- Coordinates: {response.location.latitude}, {response.location.longitude}"
            )
    except geoip2.errors.AddressNotFoundError:
        return f"Error: Address {ip_address} not found in the GeoIP database."
    except FileNotFoundError:
        return f"Error: GeoIP database not found at path: {db_path}. Please check the configuration."
    except Exception as e:
        return f"An unexpected error occurred during GeoIP lookup: {e}"