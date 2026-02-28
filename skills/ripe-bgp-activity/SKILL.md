---
name: ripe-bgp-activity
description: "Returns the number of BGP updates seen for a resource."
metadata: {"category": "network"}
---

# RIPE BGP Update Activity
This skill queries the RIPE Stat API to returns the number of BGP updates seen over time. Results are aggregated in time intervals, the lenght of which is determined by the input parameters.


## Input
- resource: The resource (ASN, prefix, IP) to look up.
- starttime: The start time for the query (ISO 8601 or unix timestamp).
- endtime: The end time for the query (ISO 8601 or unix timestamp).

## Execution
```python
import requests

def execute(resource, starttime=None, endtime=None):
    url = "https://stat.ripe.net/data/bgp-update-activity/data.json"
    params = {"resource": resource}
    if starttime:
        params["starttime"] = starttime
    if endtime:
        params["endtime"] = endtime

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("data", {})
```

## Output

A JSON object containing:
- `updates`: List with BGP update activity samples. Each sample contains:
    - `starttime`: The beginning of each sample.
    - `announcements`: The number of announcements in this sample.
    - `withdrawals`: The number of withdrawals in this sample.
- `sampling_period`: The duration in seconds of each sample interval.
- `sampling_period_human`: A human-readable version of the sampling period.
- `query_starttime`: The starttime the query covers.
- `query_endtime`: The endtime the query covers.
- `resource`: Defines the resource used for the query.
- `resource_type`: The detected type of the queried resource ("asn", 4 or 6).
- `max_samples`: The maximum number of samples returned.
