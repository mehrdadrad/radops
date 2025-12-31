from langchain_core.tools import tool
from .client import get_nautobot_client

@tool
def nautobot__get_ip_by_host(hostname: str) -> str:
    """
    Retrieve IP addresses associated with a hostname (DNS name).
    
    Args:
        hostname: The hostname to search for.
    """
    nb = get_nautobot_client()
    ips = nb.ipam.ip_addresses.filter(dns_name__icontains=hostname)
    
    if not ips:
        return f"No IP addresses found for hostname '{hostname}'."
    
    results = []
    for ip in ips:
        results.append({
            "address": str(ip.address),
            "status": str(ip.status),
            "dns_name": ip.dns_name,
            "description": ip.description,
        })
    return str(results)

@tool
def nautobot__get_prefix_details(prefix: str) -> str:
    """
    Retrieve details about an IP prefix.
    
    Args:
        prefix: The prefix (e.g., '10.0.0.0/24').
    """
    nb = get_nautobot_client()
    prefixes = nb.ipam.prefixes.filter(prefix=prefix)
    
    if not prefixes:
        return f"Prefix '{prefix}' not found."
    
    p = prefixes[0]
    info = {
        "prefix": str(p.prefix),
        "status": str(p.status),
        "site": p.site.name if getattr(p, "site", None) else None,
        "vlan": p.vlan.name if getattr(p, "vlan", None) else None,
        "role": p.role.name if getattr(p, "role", None) else None,
        "description": p.description,
    }
    return str(info)