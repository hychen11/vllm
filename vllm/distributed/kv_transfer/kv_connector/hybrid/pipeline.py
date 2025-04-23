from typing import Optional, Tuple, Union, List

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.hybrid.config import (
    HybridKVTransferConfig, P2PConfig, OffloadConfig)
from vllm.distributed.kv_transfer.kv_connector.hybrid.factory import HybridKVTransferFactory
from vllm.distributed.kv_transfer.kv_connector.hybrid.manager import HybridKVCacheManager
from vllm.distributed.kv_transfer.kv_connector.hybrid.optimized_transfer import OptimizedKVTransfer
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

class HybridKVPipeline:
    """Main pipeline for hybrid KV cache operations."""
    
    def __init__(self, 
                 config: VllmConfig,
                 is_distributed: bool = False,
                 rank: int = 0,
                 local_rank: int = 0):
        """Initialize the pipeline.
        
        Args:
            config: The vLLM config
            is_distributed: Whether running in distributed mode
            rank: Global rank (only used in distributed mode)
            local_rank: Local rank (only used in distributed mode)
        """
        self.config = config
        self.is_distributed = is_distributed
        
        # Create hybrid config
        self.hybrid_config = HybridKVTransferConfig()
        
        # For single GPU, we only enable offload
        if not is_distributed:
            self.hybrid_config.p2p_config.enabled = False
            self.hybrid_config.offload_config.enabled = True
            self.hybrid_config.offload_config.storage_path = "./kv_cache"
            
        # Create components
        self.hybrid_manager, self.optimized_transfer = HybridKVTransferFactory.create_hybrid_manager(
            rank, local_rank, config)
            
    def process_request(self,
                       model_executable: torch.nn.Module,
                       model_input: "ModelInputForGPUWithSamplingMetadata",
                       kv_caches: List[torch.Tensor]) -> Tuple[bool, Optional[Union[torch.Tensor, IntermediateTensors]]]:
        """Process a request with hybrid KV cache.
        
        Args:
            model_executable: The model executable
            model_input: The model input
            kv_caches: The KV caches
            
        Returns:
            Tuple of (bypass_model_exec, hidden_states)
        """
        return self.optimized_transfer.process_request(
            model_executable, model_input, kv_caches)
            
    def save_cache(self,
                  model_executable: torch.nn.Module,
                  model_input: "ModelInputForGPUWithSamplingMetadata",
                  kv_caches: List[torch.Tensor],
                  hidden_states: Union[torch.Tensor, IntermediateTensors]) -> None:
        """Save computed KV cache.
        
        Args:
            model_executable: The model executable
            model_input: The model input
            kv_caches: The KV caches
            hidden_states: The hidden states
        """
        self.optimized_transfer.save_cache(
            model_executable, model_input, kv_caches, hidden_states)
            
    def close(self) -> None:
        """Close the pipeline."""
        self.optimized_transfer.close() 