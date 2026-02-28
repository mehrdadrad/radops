---
name: ripe-bgp-state
description: "Fetches global BGP state data from RIPE NCC for a specific resource."
metadata: '{"category":"network"}'
---

# RIPE BGP State Lookup

This skill queries the RIPE Stat API to retrieve global BGP state information for a specific IP prefix or ASN.

## When to Use

- To verify global routing visibility of a prefix.
- To debug BGP path attributes and propagation.
- To check for potential route leaks or hijacks.

## Execution
Set the `resource` variable in the `variables` argument to the IP prefix or ASN you wish to query (e.g., `1.1.1.0/24`).

```python
import requests
import json
import sys

resource = locals().get("resource") or (sys.argv[1] if len(sys.argv) > 1 else None)

if resource:
    response = requests.get("https://stat.ripe.net/data/bgp-state/data.json", params={"resource": resource})
    print(json.dumps(response.json(), indent=2))
else:
    print("Error: The 'resource' variable is required. Please provide it in the 'variables' argument.")
```

## Input

- `resource`: The IP prefix or ASN to query (e.g., `1.1.1.0/24` or `AS3333`).

## Output

A JSON object containing:
- `status`: Request status.
- `data`: Contains `bgp_state` list with route details (path, source_id, target_prefix, etc.).