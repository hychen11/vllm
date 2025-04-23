from typing import Dict, List, Optional, Tuple
import torch
from dataclasses import dataclass
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock, BlockHashType

logger = init_logger(__name__)

@dataclass
class TrieNode:
    """Trie node for KV cache management."""
    token_id: int
    kv_cache: Optional[KVCacheBlock] = None
    children: Dict[int, 'TrieNode'] = None
    ref_count: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
            
    def add_child(self, token_id: int) -> 'TrieNode':
        if token_id not in self.children:
            self.children[token_id] = TrieNode(token_id)
        return self.children[token_id]
        
    def get_child(self, token_id: int) -> Optional['TrieNode']:
        return self.children.get(token_id)
        
    def increment_ref(self):
        self.ref_count += 1
        
    def decrement_ref(self):
        self.ref_count -= 1
        return self.ref_count == 0

class TrieKVCacheManager:
    """KV cache manager using Trie tree structure."""
    
    def __init__(self, block_size: int = 64):
        self.root = TrieNode(-1)  # Root node with dummy token
        self.block_size = block_size
        self.token_to_node: Dict[int, TrieNode] = {}
        
    def insert_sequence(self, token_ids: List[int], kv_cache: KVCacheBlock) -> None:
        """Insert a sequence of tokens and its KV cache into the trie.
        
        Args:
            token_ids: List of token IDs
            kv_cache: The KV cache block for this sequence
        """
        current = self.root
        for token_id in token_ids:
            current = current.add_child(token_id)
            current.increment_ref()
            self.token_to_node[token_id] = current
            
        current.kv_cache = kv_cache
        
    def find_longest_prefix(self, token_ids: List[int]) -> Tuple[List[int], Optional[KVCacheBlock]]:
        """Find the longest prefix match in the trie.
        
        Args:
            token_ids: List of token IDs to match
            
        Returns:
            Tuple of (matched_prefix, kv_cache)
        """
        current = self.root
        matched_prefix = []
        
        for token_id in token_ids:
            child = current.get_child(token_id)
            if child is None:
                break
            current = child
            matched_prefix.append(token_id)
            
        return matched_prefix, current.kv_cache
        
    def remove_sequence(self, token_ids: List[int]) -> None:
        """Remove a sequence from the trie.
        
        Args:
            token_ids: List of token IDs to remove
        """
        current = self.root
        for token_id in token_ids:
            child = current.get_child(token_id)
            if child is None:
                return
            if child.decrement_ref():
                # If ref count is 0, remove the node
                del current.children[token_id]
                if token_id in self.token_to_node:
                    del self.token_to_node[token_id]
            current = child
            
    def get_shared_prefixes(self) -> List[List[int]]:
        """Get all shared prefixes in the trie.
        
        Returns:
            List of shared prefixes (each prefix is a list of token IDs)
        """
        shared_prefixes = []
        
        def traverse(node: TrieNode, prefix: List[int]):
            if node.ref_count > 1:
                shared_prefixes.append(prefix)
            for token_id, child in node.children.items():
                traverse(child, prefix + [token_id])
                
        traverse(self.root, [])
        return shared_prefixes 