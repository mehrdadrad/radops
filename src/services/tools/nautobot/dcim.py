from langchain_core.tools import tool
from .client import get_nautobot_client

@tool
def nautobot__get_location_by_name(name: str) -> str:
    """
    Retrieve location details by name.
    
    Args:
        name: The name of the location.
    """
    nb = get_nautobot_client()
    try:
        locations = nb.dcim.locations.filter(name=name)
    except AttributeError:
        return "Error: 'locations' endpoint not available."

    if not locations:
        return f"Location '{name}' not found."
    
    loc = locations[0]
    info = {
        "name": loc.name,
        "type": loc.location_type.name if getattr(loc, "location_type", None) else None,
        "status": str(loc.status),
        "parent": loc.parent.name if loc.parent else None,
        "description": loc.description,
    }
    return str(info)

@tool
def nautobot__get_platform_by_name(name: str) -> str:
    """
    Retrieve platform details by name.
    
    Args:
        name: The name of the platform.
    """
    nb = get_nautobot_client()
    platforms = nb.dcim.platforms.filter(name=name)
    
    if not platforms:
        return f"Platform '{name}' not found."
    
    plat = platforms[0]
    info = {
        "name": plat.name,
        "manufacturer": plat.manufacturer.name if plat.manufacturer else None,
        "napalm_driver": plat.napalm_driver,
        "description": plat.description,
    }
    return str(info)