# RBAC Configuration Guide (`rbac.yaml`)

The `rbac.yaml` file implements Role-Based Access Control, defining which users exist and what tools they are permitted to use.

## Users

Maps user IDs (typically email addresses) to their settings, including roles and profile information.

```yaml
users:
  alice@example.com:
    role: "admin"
    first_name: "Alice"
    last_name: "Smith"
  bob@example.com:
    role: "operator"
    first_name: "Bob"
    last_name: "Jones"
```

## Role Permissions

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

## Dynamic Reloading

The application automatically watches `rbac.yaml` for changes. You can add new users or modify permissions without restarting the server.

By default, the file is checked every 60 seconds. To change this frequency, add the `reload_interval_seconds` setting to the top level of your configuration:

```yaml
reload_interval_seconds: 30
```

If the updated configuration contains errors (e.g., invalid YAML or incorrect data types), the reload will be skipped, an error will be logged, and the previous configuration will remain active.
```