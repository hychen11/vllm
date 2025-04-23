import pytest
import torch
from vllm.v1.core.trie_kv_cache import TrieKVCacheManager, TrieNode
from vllm.v1.core.kv_cache_utils import KVCacheBlock, BlockHashType
from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec

def make_kv_cache_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        tensors={},
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, torch.float32,
                                               False))
        ],
    )

def test_trie_basic_operations():
    """Test basic Trie operations."""
    trie = TrieKVCacheManager(block_size=4)
    
    # Test insert
    token_ids = [1, 2, 3, 4]
    block = KVCacheBlock(block_id=1)
    trie.insert_sequence(token_ids, block)
    
    # Test find
    matched_prefix, found_block = trie.find_longest_prefix(token_ids)
    assert matched_prefix == token_ids
    assert found_block == block
    
    # Test remove
    trie.remove_sequence(token_ids)
    matched_prefix, found_block = trie.find_longest_prefix(token_ids)
    assert not matched_prefix
    assert found_block is None

def test_trie_shared_prefixes():
    """Test shared prefix handling."""
    trie = TrieKVCacheManager(block_size=4)
    
    # Insert two sequences with shared prefix
    prefix = [1, 2, 3]
    seq1 = prefix + [4]
    seq2 = prefix + [5]
    
    block1 = KVCacheBlock(block_id=1)
    block2 = KVCacheBlock(block_id=2)
    
    trie.insert_sequence(seq1, block1)
    trie.insert_sequence(seq2, block2)
    
    # Test shared prefix
    shared_prefixes = trie.get_shared_prefixes()
    assert len(shared_prefixes) == 1
    assert shared_prefixes[0] == prefix
    
    # Test reference counting
    trie.remove_sequence(seq1)
    shared_prefixes = trie.get_shared_prefixes()
    assert len(shared_prefixes) == 1  # Prefix still shared
    
    trie.remove_sequence(seq2)
    shared_prefixes = trie.get_shared_prefixes()
    assert not shared_prefixes  # No more shared prefixes

def test_trie_integration():
    """Test Trie integration with KVCacheManager."""
    manager = KVCacheManager(
        make_kv_cache_config(4, 10),
        max_model_len=100,
        enable_caching=True,
        use_trie=True
    )
    
    # Test with shared prefix
    prefix = [1, 2, 3]
    seq1 = prefix + [4]
    seq2 = prefix + [5]
    
    req1 = Request(
        request_id="1",
        prompt=None,
        prompt_token_ids=seq1,
        sampling_params=None,
        eos_token_id=100,
        arrival_time=0
    )
    
    req2 = Request(
        request_id="2",
        prompt=None,
        prompt_token_ids=seq2,
        sampling_params=None,
        eos_token_id=100,
        arrival_time=0
    )
    
    # Process first request
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req1)
    assert not computed_blocks
    assert num_computed_tokens == 0
    
    blocks = manager.allocate_slots(req1, len(seq1), computed_blocks)
    assert len(blocks) == 2  # One full block + one partial
    
    # Process second request
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req2)
    assert len(computed_blocks) == 1  # Should find shared prefix
    assert num_computed_tokens == 3  # Length of shared prefix
    
    # Cleanup
    manager.free(req1)
    manager.free(req2)

def test_trie_performance():
    """Test Trie performance with large sequences."""
    trie = TrieKVCacheManager(block_size=4)
    
    # Generate test data
    num_sequences = 100
    sequence_length = 20
    shared_prefix_length = 10
    
    # Create sequences with shared prefix
    prefix = list(range(shared_prefix_length))
    sequences = []
    for i in range(num_sequences):
        seq = prefix + [i] * (sequence_length - shared_prefix_length)
        sequences.append(seq)
    
    # Test insert performance
    import time
    start_time = time.time()
    for i, seq in enumerate(sequences):
        block = KVCacheBlock(block_id=i)
        trie.insert_sequence(seq, block)
    insert_time = time.time() - start_time
    
    # Test find performance
    start_time = time.time()
    for seq in sequences:
        matched_prefix, _ = trie.find_longest_prefix(seq)
        assert len(matched_prefix) >= shared_prefix_length
    find_time = time.time() - start_time
    
    # Test remove performance
    start_time = time.time()
    for seq in sequences:
        trie.remove_sequence(seq)
    remove_time = time.time() - start_time
    
    # Print performance metrics
    print(f"\nPerformance metrics:")
    print(f"Insert time: {insert_time:.3f}s")
    print(f"Find time: {find_time:.3f}s")
    print(f"Remove time: {remove_time:.3f}s")
    
    # Basic performance requirements
    assert insert_time < 1.0  # Should be fast
    assert find_time < 0.5   # Should be very fast
    assert remove_time < 1.0  # Should be fast 