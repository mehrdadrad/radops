# RBAC Configuration Guide (`rbac.yaml`)

The `rbac.yaml` file implements Role-Based Access Control, defining which users exist and what tools they are permitted to use.

## 1. Users

Maps usernames to roles. The username corresponds to the ID passed during the chat session initialization.

```yaml
users:
  alice: "admin"
  bob: "operator"
  charlie: "viewer"
```

## 2. Role Permissions

Defines the list of allowed tools for each role.

*   **Wildcards**: Not currently supported; tools must be listed explicitly.
*   **System Tools**: Some internal tools may be available to all users depending on system configuration, but sensitive tools should be restricted here.

### Example Configuration

```yaml
role_permissions:
  # Read-only access
  viewer:
    - "get_geoip_location"
    - "check_host_ping"
    - "lookup_peeringdb"

  # Operational access
  operator:
    - "get_geoip_location"
    - "check_host_ping"
    - "lookup_peeringdb"
    - "restart_service"
    - "get_device_config"

  # Full access
  admin:
    - "create_jira_ticket"
    - "set_user_secrets"
    - "delete_conversation_history"
    - "manage_vector_store"
```