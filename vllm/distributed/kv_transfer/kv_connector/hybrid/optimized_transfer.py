from typing import Optional, Tuple, Union, List

import torch

from vllm.distributed.kv_transfer.kv_connector.hybrid.manager import HybridKVCacheManager
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

class OptimizedKVTransfer:
    """Optimized KV cache transfer implementation."""
    
    def __init__(self, hybrid_manager: HybridKVCacheManager):
        self.hybrid_manager = hybrid_manager
        
    def process_request(self,
                       model_executable: torch.nn.Module,
                       model_input: "ModelInputForGPUWithSamplingMetadata",
                       kv_caches: List[torch.Tensor]) -> Tuple[bool, Optional[Union[torch.Tensor, IntermediateTensors]]]:
        """Process a request with optimized KV cache transfer.
        
        Args:
            model_executable: The model executable
            model_input: The model input
            kv_caches: The KV caches
            
        Returns:
            Tuple of (bypass_model_exec, hidden_states)
        """
        # 1. Check for partial cache hit and prefill missing parts
        cache_hit, hidden_states = self.hybrid_manager.check_cache_hit(
            model_executable, model_input, kv_caches)
            
        if cache_hit:
            logger.debug("Cache hit, bypassing model execution")
            return True, hidden_states
            
        # 2. Execute model to compute missing parts
        logger.debug("Cache miss, executing model")
        return False, None
        
    def save_cache(self,
                  model_executable: torch.nn.Module,
                  model_input: "ModelInputForGPUWithSamplingMetadata",
                  kv_caches: List[torch.Tensor],
                  hidden_states: Union[torch.Tensor, IntermediateTensors]) -> None:
        """Save computed KV cache to both P2P and offload connectors.
        
        Args:
            model_executable: The model executable
            model_input: The model input
            kv_caches: The KV caches
            hidden_states: The hidden states
        """
        self.hybrid_manager.transfer_cache(
            model_executable, model_input, kv_caches, hidden_states)
            
    def close(self) -> None:
        """Close the hybrid manager."""
        self.hybrid_manager.close() 