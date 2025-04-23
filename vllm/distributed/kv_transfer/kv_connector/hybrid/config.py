from dataclasses import dataclass
from typing import Optional, Dict, Any

from vllm.distributed.kv_transfer.kv_connector.hybrid.types import KVConnectorType

@dataclass
class P2PConfig:
    """Configuration for P2P KV cache transfer."""
    enabled: bool = True
    buffer_size: int = 1e9  # 1GB
    ip: str = "127.0.0.1"
    port: int = 14579
    extra_config: Dict[str, Any] = None

@dataclass
class OffloadConfig:
    """Configuration for offload KV cache storage."""
    enabled: bool = True
    storage_path: Optional[str] = None
    max_size: int = 10e9  # 10GB
    extra_config: Dict[str, Any] = None

@dataclass
class HybridKVTransferConfig:
    """Configuration for hybrid KV cache transfer."""
    p2p_config: P2PConfig = P2PConfig()
    offload_config: OffloadConfig = OffloadConfig()
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.p2p_config.enabled and not self.offload_config.enabled:
            raise ValueError("At least one connector type must be enabled")
            
        if self.offload_config.enabled and not self.offload_config.storage_path:
            raise ValueError("Storage path must be specified when offload is enabled")
            
    def get_enabled_connectors(self) -> list[KVConnectorType]:
        """Get list of enabled connector types."""
        enabled = []
        if self.p2p_config.enabled:
            enabled.append(KVConnectorType.P2P)
        if self.offload_config.enabled:
            enabled.append(KVConnectorType.OFFLOAD)
        return enabled 