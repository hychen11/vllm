from enum import Enum
from typing import Optional, Dict, Type

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

class KVConnectorType(Enum):
    """Types of KV connectors supported in hybrid mode."""
    P2P = "p2p"  # Point-to-point transfer
    OFFLOAD = "offload"  # Offload to storage
    HYBRID = "hybrid"  # Hybrid mode

class KVConnectorManager:
    """Manager for multiple KV connectors."""
    
    def __init__(self):
        self.connectors: Dict[KVConnectorType, KVConnectorBase] = {}
        
    def register_connector(self, type: KVConnectorType, connector: KVConnectorBase) -> None:
        """Register a new connector.
        
        Args:
            type: The type of connector
            connector: The connector instance
        """
        if type in self.connectors:
            raise ValueError(f"Connector of type {type} already registered")
        self.connectors[type] = connector
        
    def get_connector(self, type: KVConnectorType) -> Optional[KVConnectorBase]:
        """Get a connector by type.
        
        Args:
            type: The type of connector to get
            
        Returns:
            The connector instance if found, None otherwise
        """
        return self.connectors.get(type)
        
    def close(self) -> None:
        """Close all registered connectors."""
        for connector in self.connectors.values():
            connector.close()
        self.connectors.clear() 