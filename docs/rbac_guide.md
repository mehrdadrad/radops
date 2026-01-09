# RBAC Configuration Guide (`rbac.yaml`)

The `rbac.yaml` file implements Role-Based Access Control, defining which users exist and what tools they are permitted to use.

## Authorization Flow

It is important to distinguish between the **LLM's intent** and the **System's permission**.

1.  **Authentication**: The user logs in (via OIDC, Auth0, or headers) and is identified by email (integrations like slack).
2.  **Role Lookup**: RadOps finds the user's role in `rbac.yaml` (local/OIDC/Auth0)
3.  **Enforcement**: Before executing the tool, RadOps checks if the tool is in the user's allowed list.
    *   **Allowed**: The tool runs.
    *   **Denied**: The execution is blocked, and a "Permission Denied" error is returned.

This ensures that while the LLM provides the intelligence to select tools, `rbac.yaml` provides the security to restrict them.

## Local Users

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

## Authentication

RadOps integrates with external Identity Providers (IdP) for authentication, while using the local configuration for authorization (role assignment).

### 1. Local Users (Authorization)

Defining users in `rbac.yaml` (as shown in the Local Users section) is the mechanism for **Authorization**. It maps an identity (email) to a Role. This applies regardless of how the user is authenticated (OIDC, Slack, etc.).

### 2. OpenID Connect (OIDC)

OpenID Connect is a simple identity layer on top of the OAuth 2.0 protocol. It allows clients to verify the identity of the end-user based on the authentication performed by an Authorization Server.

**Supported Vendors:**
Many identity providers support OIDC, including:
*   **Google Identity**
*   **Microsoft Azure Active Directory (Entra ID)**
*   **Okta**
*   **Keycloak**
*   **Amazon Cognito**

### 3. Auth0

Auth0 is a specialized identity platform that simplifies OIDC implementation.

**Configuration Example:**

```yaml
auth0:
  enabled: true
  domain: "your-tenant.auth0.com"
  client_id: "YOUR_CLIENT_ID"
  client_secret: "vault:auth#client_secret"
  role_source: native
  cache_ttl_seconds: 600
```