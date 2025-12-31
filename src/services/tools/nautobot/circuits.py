from langchain_core.tools import tool
from .client import get_nautobot_client

@tool
def nautobot__get_circuit_by_id(circuit_id: str) -> str:
    """
    Retrieve circuit information by its Circuit ID (CID).
    
    Args:
        circuit_id: The circuit ID to search for (e.g., '123-456').
    """
    nb = get_nautobot_client()
    circuits = nb.circuits.circuits.filter(cid=circuit_id)
    
    if not circuits:
        return f"Circuit with CID '{circuit_id}' not found."
    
    circuit = circuits[0]
    info = {
        "cid": circuit.cid,
        "provider": circuit.provider.name if getattr(circuit, "provider", None) else None,
        "type": circuit.type.name if getattr(circuit, "type", None) else None,
        "status": str(circuit.status),
        "description": circuit.description,
    }
    return str(info)