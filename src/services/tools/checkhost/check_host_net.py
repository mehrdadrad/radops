import asyncio
import json
from typing import List, Optional

import httpx
from langchain_core.tools import tool

API_BASE_URL = "https://check-host.net"
DEFAULT_HEADERS = {"Accept": "application/json"}


async def _request_check(check_type: str, host: str, max_nodes: Optional[int] = None, nodes: Optional[List[str]] = None) -> str:
    """Initiates a check on check-host.net and returns the request ID."""
    params = {"host": host}
    if max_nodes:
        params["max_nodes"] = str(max_nodes)
    
    # The API expects multiple 'node' query parameters, not a single comma-separated list
    node_params = [("node", node) for node in nodes] if nodes else []

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{API_BASE_URL}/check-{check_type}",
                params=list(params.items()) + node_params,
                headers=DEFAULT_HEADERS,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if data.get("ok"):
                return data["request_id"]
            else:
                raise Exception(f"API request failed: {data.get('error', 'Unknown error')}")
        except httpx.RequestError as e:
            raise Exception(f"HTTP request error: {e}")


async def _poll_for_results(request_id: str, timeout: int = 45, poll_interval: int = 5) -> dict:
    """Polls for the check results until they are complete or a timeout occurs."""
    elapsed_time = 0
    async with httpx.AsyncClient() as client:
        while elapsed_time < timeout:
            try:
                response = await client.get(f"{API_BASE_URL}/check-result/{request_id}", headers=DEFAULT_HEADERS, timeout=10)
                response.raise_for_status()
                results = response.json()
                # The check is complete if no node has a 'null' result
                if all(value is not None for value in results.values()):
                    return results
            except httpx.RequestError as e:
                # Continue polling even if one request fails
                print(f"Polling error for {request_id}: {e}")
            
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval
    raise Exception("Polling for results timed out.")


async def _run_check(check_type: str, host: str, max_nodes: Optional[int] = 3, nodes: Optional[List[str]] = None) -> str:
    """A generic function to run a check and get results."""
    try:
        request_id = await _request_check(check_type, host, max_nodes, nodes)
        results = await _poll_for_results(request_id)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"An error occurred: {e}"


@tool
async def check_host_ping(host: str, max_nodes: Optional[int] = 3, nodes: Optional[List[str]] = None) -> str:
    """
    Performs a ping check from multiple geographical locations using check-host.net.

    Args:
        host: The hostname or IP address to ping.
        max_nodes: The maximum number of random nodes to use for the check. Defaults to 3.
        nodes: A specific list of node names to use for the check (e.g., ["us1.node.check-host.net"]).

    Returns:
        A JSON string with the ping results from each node.
    """
    return await _run_check("ping", host, max_nodes, nodes)


@tool
async def check_host_http(host: str, max_nodes: Optional[int] = 3, nodes: Optional[List[str]] = None) -> str:
    """
    Performs an HTTP check from multiple geographical locations using check-host.net.

    Args:
        host: The URL to check (e.g., 'https://example.com').
        max_nodes: The maximum number of random nodes to use for the check. Defaults to 3.
        nodes: A specific list of node names to use for the check.

    Returns:
        A JSON string with the HTTP check results from each node.
    """
    return await _run_check("http", host, max_nodes, nodes)


@tool
async def check_host_tcp(host: str, max_nodes: Optional[int] = 3, nodes: Optional[List[str]] = None) -> str:
    """
    Performs a TCP port connection check from multiple geographical locations using check-host.net.

    Args:
        host: The host and port to check (e.g., 'example.com:443').
        max_nodes: The maximum number of random nodes to use for the check. Defaults to 3.
        nodes: A specific list of node names to use for the check.

    Returns:
        A JSON string with the TCP connection results from each node.
    """
    return await _run_check("tcp", host, max_nodes, nodes)


@tool
async def check_host_dns(host: str, max_nodes: Optional[int] = 3, nodes: Optional[List[str]] = None) -> str:
    """
    Performs a DNS record check from multiple geographical locations. It returns a JSON object where each key is a monitoring node and the value contains the DNS 'A' (IPv4), 'AAAA' (IPv6), and 'TTL' records.
    You MUST parse this JSON and present the results to the user in a clear, formatted list. For each node, display the A records, AAAA records (if any), and TTL.

    For example, format the output like this:
    1. **at1.node.check-host.net**:
       - A Records: 74.208.236.26
       - TTL: 300
    
    2. **ca1.node.check-host.net**:
       - A Records: 74.208.236.26
       - TTL: 300


    Args:
        host: The hostname to resolve.
        max_nodes: The maximum number of random nodes to use for the check. Defaults to 3.
        nodes: A specific list of node names to use for the check.

    Returns:
        A JSON string with the DNS resolution results from each node.
    """
    return await _run_check("dns", host, max_nodes, nodes)


@tool
async def get_check_host_nodes() -> str:
    """
    Fetches the list of available monitoring nodes from check-host.net.
    Use this to find valid node names for other check-host tools.

    Returns:
        A JSON string containing a list of available node hostnames.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/nodes/hosts", headers=DEFAULT_HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Extract just the node names for clarity
            node_names = list(data.get("nodes", {}).keys())
            return json.dumps(node_names, indent=2)
        except httpx.RequestError as e:
            return f"An error occurred while fetching nodes: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"