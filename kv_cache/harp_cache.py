#!/usr/bin/env python3
"""
harp_cache.py - Hash-Addressed Radix Page Sharing (HARP-Cache) Engine
Part of the RouteCrack-KV Zero-FLOP Optimization Suite.

Provides content-addressable, zero-copy KV page sharing via POSIX shared memory (shm),
FNV-64 rolling hash chunking, and instant prefix pre-fill elimination for agentic loops.
"""

import os
import sys
import time
import mmap
import struct
import threading
from typing import List, Tuple, Dict, Optional, Any

# FNV-1a 64-bit constants
FNV_OFFSET_BASIS = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
MASK_64 = 0xffffffffffffffff

def fnv1a_64(data: bytes, seed: int = FNV_OFFSET_BASIS) -> int:
    """Computes high-speed 64-bit FNV-1a hash over raw byte buffer."""
    h = seed
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK_64
    return h

def hash_token_block(tokens: List[int]) -> int:
    """Hashes a contiguous sequence of token IDs using FNV-1a."""
    raw = struct.pack(f"{len(tokens)}i", *tokens)
    return fnv1a_64(raw)

class HARPPage:
    """Metadata descriptor for a single 32-token KV cache memory page."""
    __slots__ = ("page_id", "page_hash", "token_count", "tokens", "entropy_score", "shm_offset", "access_count", "created_at")
    
    def __init__(self, page_id: int, page_hash: int, tokens: List[int], entropy_score: float, shm_offset: int):
        self.page_id = page_id
        self.page_hash = page_hash
        self.token_count = len(tokens)
        self.tokens = tokens
        self.entropy_score = entropy_score
        self.shm_offset = shm_offset
        self.access_count = 1
        self.created_at = time.time()

class HARPCacheEngine:
    """
    High-Performance Zero-Copy Radix Page Sharing Engine.
    Manages content-addressable KV pages in shared memory for zero-prefill agentic loops.
    """
    PAGE_TOKENS = 32                  # 32 tokens per physical memory page
    PAGE_KV_BYTES = 32 * 196608 // 48  # Simulated KV footprint per 32-token page in Q4_0/FP16 mixed mode (~131 KB)
    MAX_PAGES = 4096                  # 4096 pages = 131,072 tokens capacity (~512 MB shm)
    SHM_FILE_PATH = "/dev/shm/harp_kv_cache.bin"

    def __init__(self, shm_path: str = SHM_FILE_PATH, max_pages: int = MAX_PAGES):
        self.shm_path = shm_path
        self.max_pages = max_pages
        self.total_shm_bytes = self.max_pages * self.PAGE_KV_BYTES
        self.lock = threading.Lock()
        
        # Hash lookup index: page_hash (uint64) -> HARPPage
        self.hash_to_page: Dict[int, HARPPage] = {}
        
        # Radix tree root: node maps token_id -> next_node, and leaf maps to page
        self.radix_root: Dict[Any, Any] = {}
        
        # Page allocator
        self.allocated_pages = 0
        self._init_shm()

    def _init_shm(self):
        """Initializes POSIX shared memory backing file with zero-copy mmap."""
        try:
            # Check or create file
            if not os.path.exists(self.shm_path) or os.path.getsize(self.shm_path) != self.total_shm_bytes:
                with open(self.shm_path, "wb") as f:
                    # Sparse file allocation
                    f.seek(self.total_shm_bytes - 1)
                    f.write(b"\0")
            
            # Memory map
            self.shm_file = open(self.shm_path, "r+b")
            self.shm_mmap = mmap.mmap(self.shm_file.fileno(), self.total_shm_bytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            self.shm_active = True
        except Exception as e:
            # Fallback to anonymous heap memory if /dev/shm access is restricted
            self.shm_mmap = bytearray(self.total_shm_bytes)
            self.shm_active = False

    def close(self):
        """Cleans up memory mappings and files."""
        with self.lock:
            if hasattr(self, 'shm_mmap') and hasattr(self.shm_mmap, 'close'):
                self.shm_mmap.close()
            if hasattr(self, 'shm_file') and not self.shm_file.closed:
                self.shm_file.close()

    def calculate_entropy(self, tokens: List[int]) -> float:
        """Estimates token block entropy using RouteCrack zero-FLOP heuristics."""
        if not tokens:
            return 0.0
        unique_tokens = len(set(tokens))
        return float(unique_tokens) / float(len(tokens))

    def resolve_prefix(self, token_stream: List[int]) -> Tuple[int, List[HARPPage], List[int]]:
        """
        Longest Common Prefix (LCP) Resolver.
        Scans incoming token stream block-by-block (32 tokens) against HARP-Cache.
        Returns:
            (cached_tokens_count, matched_pages, remaining_suffix_tokens)
        """
        t0 = time.perf_counter_ns()
        matched_pages: List[HARPPage] = []
        n_tokens = len(token_stream)
        curr_offset = 0
        
        with self.lock:
            while curr_offset + self.PAGE_TOKENS <= n_tokens:
                chunk = token_stream[curr_offset : curr_offset + self.PAGE_TOKENS]
                h = hash_token_block(chunk)
                
                page = self.hash_to_page.get(h)
                if page is not None and page.tokens == chunk:
                    page.access_count += 1
                    matched_pages.append(page)
                    curr_offset += self.PAGE_TOKENS
                else:
                    break
        
        cached_count = len(matched_pages) * self.PAGE_TOKENS
        remaining_suffix = token_stream[cached_count:]
        lookup_ns = time.perf_counter_ns() - t0
        return cached_count, matched_pages, remaining_suffix

    def store_prefix(self, token_stream: List[int], simulated_kv_data: Optional[bytes] = None) -> int:
        """
        Chunks and stores an immutable prefix into HARP-Cache in shared memory.
        Returns number of newly stored pages.
        """
        new_pages = 0
        n_tokens = len(token_stream)
        curr_offset = 0
        
        with self.lock:
            while curr_offset + self.PAGE_TOKENS <= n_tokens:
                chunk = token_stream[curr_offset : curr_offset + self.PAGE_TOKENS]
                h = hash_token_block(chunk)
                
                if h not in self.hash_to_page:
                    if self.allocated_pages >= self.max_pages:
                        # Simple LRU eviction
                        oldest_hash = min(self.hash_to_page.keys(), key=lambda k: self.hash_to_page[k].created_at)
                        del self.hash_to_page[oldest_hash]
                    
                    page_id = self.allocated_pages
                    shm_offset = page_id * self.PAGE_KV_BYTES
                    entropy = self.calculate_entropy(chunk)
                    
                    # Create page
                    page = HARPPage(
                        page_id=page_id,
                        page_hash=h,
                        tokens=chunk,
                        entropy_score=entropy,
                        shm_offset=shm_offset
                    )
                    
                    # Write simulated KV data to shared memory
                    if simulated_kv_data and len(simulated_kv_data) >= self.PAGE_KV_BYTES:
                        self.shm_mmap[shm_offset : shm_offset + self.PAGE_KV_BYTES] = simulated_kv_data[:self.PAGE_KV_BYTES]
                    else:
                        self.shm_mmap[shm_offset : shm_offset + 32] = struct.pack("QQQQ", h, len(chunk), int(entropy*1000), page_id)
                    
                    self.hash_to_page[h] = page
                    self.allocated_pages = (self.allocated_pages + 1) % self.max_pages
                    new_pages += 1
                
                curr_offset += self.PAGE_TOKENS
                
        return new_pages

    def stats(self) -> Dict[str, Any]:
        """Returns engine runtime diagnostics."""
        with self.lock:
            cached_pages = len(self.hash_to_page)
            total_cached_tokens = cached_pages * self.PAGE_TOKENS
            total_shm_mb = (cached_pages * self.PAGE_KV_BYTES) / (1024 * 1024)
            avg_entropy = sum(p.entropy_score for p in self.hash_to_page.values()) / max(1, cached_pages)
            return {
                "active_pages": cached_pages,
                "cached_tokens": total_cached_tokens,
                "shm_footprint_mb": total_shm_mb,
                "avg_entropy": avg_entropy,
                "shm_active": self.shm_active,
                "shm_path": self.shm_path
            }

if __name__ == "__main__":
    print("[*] HARP-Cache Engine Module compiled successfully.")
