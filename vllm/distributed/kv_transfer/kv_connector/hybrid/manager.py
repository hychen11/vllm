from typing import Optional, Tuple, Union, List

import torch

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.hybrid.types import KVConnectorType, KVConnectorManager
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

class HybridKVCacheManager:
    """Manager for hybrid KV cache operations."""
    
    def __init__(self):
        self.connector_manager = KVConnectorManager()
        
    def register_connectors(self, p2p_connector: KVConnectorBase, 
                          offload_connector: KVConnectorBase) -> None:
        """Register P2P and offload connectors.
        
        Args:
            p2p_connector: The P2P connector instance
            offload_connector: The offload connector instance
        """
        self.connector_manager.register_connector(KVConnectorType.P2P, p2p_connector)
        self.connector_manager.register_connector(KVConnectorType.OFFLOAD, offload_connector)
        
    def check_cache_hit(self, 
                       model_executable: torch.nn.Module,
                       model_input: "ModelInputForGPUWithSamplingMetadata",
                       kv_caches: List[torch.Tensor]) -> Tuple[bool, Optional[Union[torch.Tensor, IntermediateTensors]]]:
        """Check if the request hits cache in either P2P or offload connector.
        
        Args:
            model_executable: The model executable
            model_input: The model input
            kv_caches: The KV caches
            
        Returns:
            Tuple of (cache_hit, hidden_states)
        """
        # First check P2P connector
        p2p_connector = self.connector_manager.get_connector(KVConnectorType.P2P)
        if p2p_connector:
            hidden_states, bypass_model_exec, _ = p2p_connector.recv_kv_caches_and_hidden_states(
                model_executable, model_input, kv_caches)
            if bypass_model_exec:
                return True, hidden_states
                
        # Then check offload connector
        offload_connector = self.connector_manager.get_connector(KVConnectorType.OFFLOAD)
        if offload_connector:
            hidden_states, bypass_model_exec, _ = offload_connector.recv_kv_caches_and_hidden_states(
                model_executable, model_input, kv_caches)
            if bypass_model_exec:
                return True, hidden_states
                
        return False, None
        
    def transfer_cache(self,
                      model_executable: torch.nn.Module,
                      model_input: "ModelInputForGPUWithSamplingMetadata",
                      kv_caches: List[torch.Tensor],
                      hidden_states: Union[torch.Tensor, IntermediateTensors]) -> None:
        """Transfer KV cache using both P2P and offload connectors.
        
        Args:
            model_executable: The model executable
            model_input: The model input
            kv_caches: The KV caches
            hidden_states: The hidden states
        """
        # Send to P2P connector
        p2p_connector = self.connector_manager.get_connector(KVConnectorType.P2P)
        if p2p_connector:
            p2p_connector.send_kv_caches_and_hidden_states(
                model_executable, model_input, kv_caches, hidden_states)
                
        # Send to offload connector
        offload_connector = self.connector_manager.get_connector(KVConnectorType.OFFLOAD)
        if offload_connector:
            offload_connector.send_kv_caches_and_hidden_states(
                model_executable, model_input, kv_caches, hidden_states)
                
    def close(self) -> None:
        """Close all connectors."""
        self.connector_manager.close() 