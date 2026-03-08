---
name: workflow-host-health-check
description: "Daily System Host Health Check"
metadata: '{"category":"network"}'
---

## Execution

1. **Check DNS**
- **Instruction:** Check the A record for the host
- **Condition** Skip to "Summarize Findings" if the DNS report does not resolve to an IP address

2. **Check HTTP Status Code**
- **Instruction:** HTTP check for the host

3. **Ping the resolved IP address**
- **Instruction:** Ping the resolved IP address of host, if it more than 50ms then it is warning

4. **Summarize Findings**
- **Instruction:** Review the network outputs generate a summary report and rate the findings for the health check (healthy, warning, critical).