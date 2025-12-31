from langchain_core.tools import tool
from .client import get_nautobot_client

@tool
def nautobot__get_device_details(name: str) -> str:
    """
    Retrieve detailed information about a device.
    
    Args:
        name: The name of the device.
    """
    nb = get_nautobot_client()
    try:
        device = nb.dcim.devices.get(name=name)
    except Exception:
        return f"Error retrieving device '{name}'."

    if not device:
        return f"Device '{name}' not found."

    info = {
        "name": device.name,
        "role": device.device_role.name if getattr(device, "device_role", None) else None,
        "type": device.device_type.model if getattr(device, "device_type", None) else None,
        "site": device.site.name if getattr(device, "site", None) else None,
        "status": str(device.status),
        "primary_ip": str(device.primary_ip) if getattr(device, "primary_ip", None) else None,
        "serial": getattr(device, "serial", None),
        "platform": device.platform.name if getattr(device, "platform", None) else None,
    }
    return str(info)

@tool
def nautobot__get_device_interfaces(name: str) -> str:
    """
    Retrieve interfaces for a specific device.
    
    Args:
        name: The name of the device.
    """
    nb = get_nautobot_client()
    try:
        device = nb.dcim.devices.get(name=name)
    except Exception:
        return f"Error retrieving device '{name}'."

    if not device:
        return f"Device '{name}' not found."

    interfaces = nb.dcim.interfaces.filter(device_id=device.id)
    
    results = [
        {"name": i.name, "type": i.type.label if i.type else "Unknown", "enabled": i.enabled, "description": i.description}
        for i in interfaces
    ]
    return str(results)