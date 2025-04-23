from typing import Optional, Tuple

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.hybrid.config import HybridKVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.hybrid.manager import HybridKVCacheManager
from vllm.distributed.kv_transfer.kv_connector.hybrid.optimized_transfer import OptimizedKVTransfer
from vllm.distributed.kv_transfer.kv_connector.hybrid.types import KVConnectorType
from vllm.logger import init_logger

logger = init_logger(__name__)

class HybridKVTransferFactory:
    """Factory for creating hybrid KV transfer components."""
    
    @staticmethod
    def create_hybrid_manager(rank: int, local_rank: int, 
                            config: VllmConfig) -> Tuple[HybridKVCacheManager, OptimizedKVTransfer]:
        """Create a hybrid KV cache manager and optimized transfer.
        
        Args:
            rank: The global rank
            local_rank: The local rank
            config: The vLLM config
            
        Returns:
            Tuple of (hybrid_manager, optimized_transfer)
        """
        # Create hybrid config
        hybrid_config = HybridKVTransferConfig()
        hybrid_config.validate()
        
        # Create hybrid manager
        hybrid_manager = HybridKVCacheManager()
        
        # Create and register connectors
        for connector_type in hybrid_config.get_enabled_connectors():
            if connector_type == KVConnectorType.P2P:
                p2p_config = hybrid_config.p2p_config
                connector = KVConnectorFactory.create_connector(
                    rank, local_rank, config)
                hybrid_manager.register_connectors(connector, None)
                
            elif connector_type == KVConnectorType.OFFLOAD:
                offload_config = hybrid_config.offload_config
                connector = KVConnectorFactory.create_connector(
                    rank, local_rank, config)
                hybrid_manager.register_connectors(None, connector)
                
        # Create optimized transfer
        optimized_transfer = OptimizedKVTransfer(hybrid_manager)
        
        return hybrid_manager, optimized_transfer 