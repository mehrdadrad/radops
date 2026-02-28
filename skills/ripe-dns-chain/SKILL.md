---
name: ripe-dns-chain
description: "Fetches DNS chain data from RIPE NCC for a specific domain."
metadata: '{"category":"network"}'
---

# RIPE DNS Chain Lookup

This skill queries the RIPE Stat API to retrieve the DNS chain for a specific domain name.

## When to Use

- To troubleshoot DNS resolution issues.
- To verify the delegation chain of a domain.

## Execution

Set the `resource` variable in the `variables` argument to the domain name you wish to query (e.g., `www.ippacket.org`).

```python
import requests
import json
import sys

resource = locals().get("resource") or (sys.argv[1] if len(sys.argv) > 1 else None)

if resource:
    response = requests.get("https://stat.ripe.net/data/dns-chain/data.json", params={"resource": resource})
    print(json.dumps(response.json(), indent=2))
else:
    print("Error: The 'resource' variable is required. Please provide it in the 'variables' argument.")
```

## Input

- `resource`: The domain name to query (e.g., `www.ippacket.org`).

## Output

A JSON object containing:
- `status`: Request status.
- `data`: Contains `forward_nodes` and `reverse_nodes` lists with DNS resolution details.