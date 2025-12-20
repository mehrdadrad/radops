import requests
import time
import json
from langchain.tools import tool

@tool
def verizon_looking_glass(destination: str, command: str = "ping", source: str = "ASHBURN,USA", ip_version: str = "ipv4") -> str:
    """
    Runs a network command ('ping', 'trace', 'bgp') from a Verizon source to a destination.
    
    IMPORTANT: This is a two-step process.
    1. First, you MUST call the 'verizon_looking_glass_locations' tool to get the list of valid source locations.
    2. From that list, find the location string (e.g., 'ASHBURN,USA') that close to the user's request.
    3. ONLY then, call this tool using that exact string for the 'source' argument. Do not guess or invent a source location.

    Args:
        destination: The destination IP address or hostname.
        command: The command to run. Can be 'ping', 'trace', or 'bgp'. Defaults to 'ping'.
        source: The exact source location string obtained from the 'verizon_looking_glass_locations' tool (e.g., 'ASHBURN,USA').
        ip_version: The IP version to use. Can be 'ipv4' or 'ipv6'. Defaults to 'ipv4'.

    Returns:
        A JSON string containing the results of the network command if successful, or an error message if it fails.
    """
    url = "https://www.verizon.com/business/api/lg.js?cmd"

    headers = {
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'accept-language': 'en-US,en;q=0.5',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'origin': 'https://www.verizon.com',
        'priority': 'u=1, i',
        'referer': 'https://www.verizon.com/business/why-verizon/looking-glass/',
        'sec-ch-ua': '"Chromium";v="142", "Brave";v="142", "Not_A Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
    }

    # The token is likely required and might be dynamic. For now, it's hardcoded from the curl.
    data = {
        'src': source,
        'dest': destination,
        'cmd': command,
        'ipv': ip_version
       # 'token': 'IgTtsBbySj5JA5yjYnAgxq7ehVOA'
    }

    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=30)
        response.raise_for_status() 
        data = response.json()
        data['location'] = source
        if command != "trace":
            return json.dumps(data)
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}, data: {data}"
    except json.JSONDecodeError:
        return f"Error: Failed to decode JSON. The server may have returned an empty or invalid response. Status: {response.status_code}, Body: {response.text[:200]}"

    if data.get('status') == 'IN_PROGRESS' and 'requestId' in data:
        request_id = data['requestId']
        max_retries = 10  
        for i in range(max_retries):
            print(f"Polling for traceroute result... Attempt {i+1}/{max_retries}")
            time.sleep(5)  
            print("request id", request_id) 
            poll_params = {'cmd': 'trace', 'requestId': request_id}
            poll_response = requests.post(url, headers=headers, data=poll_params, timeout=30)
            poll_response.raise_for_status()
            poll_data = poll_response.json()

            if poll_data.get('status') != 'IN_PROGRESS':
                poll_data['location'] = source
                return json.dumps(poll_data) 
        
        return "Error: Traceroute timed out after 60 seconds."
    else:
        return f"Error: Unexpected initial response for traceroute: {json.dumps(poll_data)}" 
    
@tool
def verizon_looking_glass_locations() -> list[str] | str:
    """
    Fetches the list of available source locations for the Verizon Looking Glass tool.
    Returns a list of location strings (e.g., 'ASHBURN,USA') that can be used as the 'source' argument in the 'verizon_looking_glass' tool.
    """
    url = "https://www.verizon.com/business/api/lg.js?cmd"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        
        locations = []
        # The JSON is a dict where keys are continents and values are lists of location dicts.
        for continent, cities in data.items():
            if isinstance(cities, list):
                for city_info in cities:
                    if 'location' in city_info:
                        locations.append(city_info['location'])
        return locations
    except requests.exceptions.RequestException as e:
        return f"An error occurred while fetching sources: {e}"
    except ValueError: # Catches JSON decoding errors
        return "Error: Failed to decode JSON from the response."