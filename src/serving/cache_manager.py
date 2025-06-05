"""
KV-Cache management for adaptive speculative decoding
"""

import torch
from typing import Dict, List, Optional, Any
import time
import threading
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry"""
    cache_data: Dict[str, torch.Tensor]
    creation_time: float
    last_access: float
    size_bytes: int
    stage_id: int


class KVCacheManager:
    """
    Manages KV-Cache for multi-stage pipeline with dynamic stopping
    Optimized for H100 memory management
    """
    
    def __init__(
        self,
        num_stages: int = 4,
        max_cache_size_gb: float = 40.0,
        cleanup_interval: float = 300.0,  # 5 minutes
        enable_compression: bool = False
    ):
        self.num_stages = num_stages
        self.max_cache_size_bytes = int(max_cache_size_gb * (1024 ** 3))
        self.cleanup_interval = cleanup_interval
        self.enable_compression = enable_compression
        
        # Cache storage: request_id -> stage_id -> CacheEntry
        self.caches: Dict[str, Dict[int, CacheEntry]] = defaultdict(dict)
        
        # Statistics
        self.stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "current_size_bytes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cleanup_count": 0
        }
        
        # Thread lock for concurrent access
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True
        )
        self.cleanup_thread.start()
        
        logger.info(f"KVCacheManager initialized with {max_cache_size_gb}GB limit")
    
    def allocate(
        self,
        request_id: str,
        stage_id: int,
        cache_data: Dict[str, torch.Tensor]
    ) -> bool:
        """
        Allocate cache for a specific stage
        
        Args:
            request_id: Unique request identifier
            stage_id: Stage number (0-based)
            cache_data: Dictionary of cache tensors
            
        Returns:
            success: Whether allocation succeeded
        """
        with self.lock:
            try:
                # Calculate cache size
                cache_size = self._calculate_size(cache_data)
                
                # Check if we have enough space
                if not self._ensure_space(cache_size):
                    logger.warning(f"Insufficient cache space for request {request_id}")
                    return False
                
                # Create cache entry
                entry = CacheEntry(
                    cache_data=cache_data,
                    creation_time=time.time(),
                    last_access=time.time(),
                    size_bytes=cache_size,
                    stage_id=stage_id
                )
                
                # Store cache
                self.caches[request_id][stage_id] = entry
                
                # Update statistics
                self.stats["total_allocations"] += 1
                self.stats["current_size_bytes"] += cache_size
                
                logger.debug(f"Allocated {cache_size/1024**2:.1f}MB cache for "
                           f"request {request_id}, stage {stage_id}")
                
                return True
                
            except Exception as e:
                logger.error(f"Cache allocation failed: {e}")
                return False
    
    def get_cache(
        self,
        request_id: str,
        stage_id: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve cache for a specific stage
        
        Args:
            request_id: Request identifier
            stage_id: Stage number
            
        Returns:
            cache_data: Cache tensors or None if not found
        """
        with self.lock:
            if request_id in self.caches and stage_id in self.caches[request_id]:
                entry = self.caches[request_id][stage_id]
                entry.last_access = time.time()
                self.stats["cache_hits"] += 1
                
                logger.debug(f"Cache hit for request {request_id}, stage {stage_id}")
                return entry.cache_data
            else:
                self.stats["cache_misses"] += 1
                logger.debug(f"Cache miss for request {request_id}, stage {stage_id}")
                return None
    
    def truncate_at_stage(
        self,
        request_id: str,
        final_stage: int
    ):
        """
        Remove caches for stages beyond the final stage
        
        Args:
            request_id: Request identifier
            final_stage: Last stage to keep
        """
        with self.lock:
            if request_id not in self.caches:
                return
            
            stages_to_remove = [
                stage for stage in self.caches[request_id].keys()
                if stage > final_stage
            ]
            
            freed_bytes = 0
            for stage in stages_to_remove:
                entry = self.caches[request_id][stage]
                freed_bytes += entry.size_bytes
                
                # Free GPU memory
                for tensor in entry.cache_data.values():
                    if tensor.is_cuda:
                        del tensor
                
                del self.caches[request_id][stage]
            
            if freed_bytes > 0:
                self.stats["current_size_bytes"] -= freed_bytes
                self.stats["total_deallocations"] += len(stages_to_remove)
                
                logger.debug(f"Freed {freed_bytes/1024**2:.1f}MB from stages {stages_to_remove}")
                
                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def cleanup_request(
        self,
        request_id: str
    ):
        """
        Remove all caches for a completed request
        
        Args:
            request_id: Request identifier
        """
        with self.lock:
            if request_id not in self.caches:
                return
            
            freed_bytes = 0
            num_stages = len(self.caches[request_id])
            
            for stage_id, entry in self.caches[request_id].items():
                freed_bytes += entry.size_bytes
                
                # Free GPU memory
                for tensor in entry.cache_data.values():
                    if tensor.is_cuda:
                        del tensor
            
            del self.caches[request_id]
            
            self.stats["current_size_bytes"] -= freed_bytes
            self.stats["total_deallocations"] += num_stages
            
            logger.debug(f"Cleaned up request {request_id}: "
                        f"freed {freed_bytes/1024**2:.1f}MB from {num_stages} stages")
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _calculate_size(
        self,
        cache_data: Dict[str, torch.Tensor]
    ) -> int:
        """Calculate total size of cache data in bytes"""
        total_bytes = 0
        for tensor in cache_data.values():
            if isinstance(tensor, torch.Tensor):
                total_bytes += tensor.element_size() * tensor.nelement()
        return total_bytes
    
    def _ensure_space(
        self,
        required_bytes: int
    ) -> bool:
        """
        Ensure sufficient cache space by evicting old entries if needed
        
        Args:
            required_bytes: Required space in bytes
            
        Returns:
            success: Whether space was freed successfully
        """
        if self.stats["current_size_bytes"] + required_bytes <= self.max_cache_size_bytes:
            return True
        
        # Need to free space - use LRU eviction
        return self._evict_lru(required_bytes)
    
    def _evict_lru(
        self,
        required_bytes: int
    ) -> bool:
        """
        Evict least recently used cache entries
        
        Args:
            required_bytes: Bytes to free
            
        Returns:
            success: Whether enough space was freed
        """
        # Collect all entries with access times
        all_entries = []
        for request_id, stages in self.caches.items():
            for stage_id, entry in stages.items():
                all_entries.append((entry.last_access, request_id, stage_id, entry))
        
        # Sort by access time (oldest first)
        all_entries.sort(key=lambda x: x[0])
        
        freed_bytes = 0
        entries_removed = 0
        
        for _, request_id, stage_id, entry in all_entries:
            if freed_bytes >= required_bytes:
                break
            
            # Free this entry
            freed_bytes += entry.size_bytes
            entries_removed += 1
            
            # Free GPU memory
            for tensor in entry.cache_data.values():
                if tensor.is_cuda:
                    del tensor
            
            # Remove from cache
            del self.caches[request_id][stage_id]
            
            # Remove request if no stages left
            if not self.caches[request_id]:
                del self.caches[request_id]
        
        self.stats["current_size_bytes"] -= freed_bytes
        self.stats["total_deallocations"] += entries_removed
        
        logger.info(f"LRU eviction: freed {freed_bytes/1024**2:.1f}MB from {entries_removed} entries")
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return freed_bytes >= required_bytes
    
    def _periodic_cleanup(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cleanup thread error: {e}")
    
    def _cleanup_expired(
        self,
        max_age_seconds: float = 1800.0  # 30 minutes
    ):
        """Remove entries older than max_age_seconds"""
        with self.lock:
            current_time = time.time()
            expired_entries = []
            
            for request_id, stages in self.caches.items():
                for stage_id, entry in stages.items():
                    if current_time - entry.last_access > max_age_seconds:
                        expired_entries.append((request_id, stage_id))
            
            if not expired_entries:
                return
            
            freed_bytes = 0
            for request_id, stage_id in expired_entries:
                if request_id in self.caches and stage_id in self.caches[request_id]:
                    entry = self.caches[request_id][stage_id]
                    freed_bytes += entry.size_bytes
                    
                    # Free GPU memory
                    for tensor in entry.cache_data.values():
                        if tensor.is_cuda:
                            del tensor
                    
                    del self.caches[request_id][stage_id]
                    
                    # Remove request if no stages left
                    if not self.caches[request_id]:
                        del self.caches[request_id]
            
            self.stats["current_size_bytes"] -= freed_bytes
            self.stats["total_deallocations"] += len(expired_entries)
            self.stats["cleanup_count"] += 1
            
            logger.info(f"Expired cleanup: removed {len(expired_entries)} entries, "
                       f"freed {freed_bytes/1024**2:.1f}MB")
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            active_requests = len(self.caches)
            active_stages = sum(len(stages) for stages in self.caches.values())
            
            return {
                **self.stats,
                "active_requests": active_requests,
                "active_stages": active_stages,
                "current_size_mb": self.stats["current_size_bytes"] / (1024 ** 2),
                "max_size_mb": self.max_cache_size_bytes / (1024 ** 2),
                "utilization": self.stats["current_size_bytes"] / self.max_cache_size_bytes,
                "hit_rate": (
                    self.stats["cache_hits"] / 
                    (self.stats["cache_hits"] + self.stats["cache_misses"])
                    if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
                )
            }