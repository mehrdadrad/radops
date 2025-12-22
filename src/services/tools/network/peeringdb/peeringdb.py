import json
from typing import Optional

import httpx
from langchain.tools import tool

from config.tools import tool_settings as settings


@tool
async def network__get_asn_peering_info(asn: str) -> str:
    """
    Fetches detailed information about a network from PeeringDB based on its
    Autonomous System Number (ASN).

    This includes organization details, name, website, traffic volume, peering
    policy, and other network-related data.

    IMPORTANT DISPLAY INSTRUCTIONS:
    - The output will be raw JSON. Do NOT show the raw JSON to the user until user requests it.
    - Format the data pretty 
    """
    url = f"https://www.peeringdb.com/api/net?asn={asn}"
    headers = {"Authorization": "Api-Key " + settings.peeringdb.api_key}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data.get("data"):
                return json.dumps(data["data"][0])
            return f"No information found for ASN {asn}."
    except httpx.HTTPStatusError as e:
        return (
            f"API request failed with status {e.response.status_code} "
            f"for ASN {asn}."
        )
    except Exception as e:
        return f"An error occurred while fetching peering info for ASN {asn}: {e}"


@tool
async def network__get_peering_exchange_info(
    asn: str, filter_by_name: Optional[str] = None
) -> str:
    """
    Fetches a list of public Internet Exchange (IX) points for a given
    Autonomous System Number (ASN).

    This tool is useful for discovering where a network peers publicly.
    The returned information for each exchange includes the IX name, connection
    speed, and the IPv4/IPv6 addresses used for peering.
    Don't summarize the return data.

    Args:
        asn (str): The Autonomous System Number to look up (e.g., '6507').
        filter_by_name (str, optional): A name to filter the IX list by.
            For example, 'DE-CIX'. Defaults to None.
    """
    url = f"https://www.peeringdb.com/api/netixlan?asn={asn}"
    headers = {"Authorization": "Api-Key " + settings.peeringdb.api_key}
    LIMIT = 20
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data.get("data"):
                all_records = data["data"]

                # Filter by name if provided (case-insensitive)
                if filter_by_name:
                    filter_by_name_lower = filter_by_name.lower()
                    all_records = [
                        record
                        for record in all_records
                        if filter_by_name_lower in record.get("name", "").lower()
                    ]

                # Define the fields you want to keep for a more compressed output
                fields_to_keep = [
                    "name",
                    "speed",
                    "ipaddr4",
                    "ipaddr6",
                    "is_rs_peer",
                ]
                compressed_data = [
                    {key: record[key] for key in fields_to_keep if key in record}
                    for record in all_records
                ]
                if len(compressed_data) > LIMIT:
                    compressed_data = compressed_data[:LIMIT]
                return json.dumps(compressed_data)
            return f"No information found for ASN {asn}."
    except httpx.HTTPStatusError as e:
        return (
            f"API request failed with status {e.response.status_code} "
            f"for ASN {asn}."
        )
    except Exception as e:
        return (
            f"An error occurred while fetching peering exchange info "
            f"for ASN {asn}: {e}"
        )