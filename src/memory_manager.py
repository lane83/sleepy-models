#!/usr/bin/env python3
"""
Memory management system for Sleepy-Models.
Handles short-term and long-term memory storage, decay, and reinforcement.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import heapq
import threading
from filelock import FileLock


@dataclass
class Memory:
    """Data class for storing memory items."""
    id: str
    content: str
    embedding: List[float]
    token_count: int
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    consolidated: bool
    context: Dict
    relationships: Set[str]  # Set of related memory IDs


class MemoryManager:
    """Manages memory storage, retrieval, and optimization."""
    
    def __init__(self, data_dir: str = "data", knowledge_graph=None, usage_tracker=None):
        """
        Initialize the memory manager.
        
        Args:
            data_dir (str): Directory to store memory data
            knowledge_graph: Reference to knowledge graph instance
            usage_tracker: Reference to usage tracker instance
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.memories_file = self.data_dir / "memories.json"
        self.lock_file = self.data_dir / "memories.lock"
        self.file_lock = FileLock(str(self.lock_file))
        
        # System integration
        self.knowledge_graph = knowledge_graph
        self.usage_tracker = usage_tracker
        
        # Memory storage
        self.short_term_memory: Dict[str, Memory] = {}
        self.long_term_memory: Dict[str, Memory] = {}
        
        # Cache for quick lookups
        self.embedding_cache: Dict[str, List[float]] = {}
        self.importance_cache: Dict[str, float] = {}
        
        # Access tracking
        self.access_history: Dict[str, List[datetime]] = defaultdict(list)
        self.access_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Configuration
        self.config = {
            "short_term_capacity": 1000,  # memories
            "long_term_capacity": 100000,  # memories
            "consolidation_threshold": 0.7,  # importance score
            "decay_rate": 0.1,  # per day
            "access_window": 7,  # days
            "importance_threshold": 0.5,
            "max_relationships": 50  # per memory
        }
        
        self.lock = threading.Lock()
        self.load_memories()

    def load_memories(self) -> None:
        """Load memories from storage."""
        with self.file_lock:
            try:
                if self.memories_file.exists():
                    with open(self.memories_file, 'r') as f:
                        data = json.load(f)
                        
                        # Load short-term memories
                        for mem_data in data.get("short_term", []):
                            memory = self._create_memory_from_dict(mem_data)
                            self.short_term_memory[memory.id] = memory
                        
                        # Load long-term memories
                        for mem_data in data.get("long_term", []):
                            memory = self._create_memory_from_dict(mem_data)
                            self.long_term_memory[memory.id] = memory
                        
                        # Load access history
                        for mem_id, access_times in data.get("access_history", {}).items():
                            self.access_history[mem_id] = [
                                datetime.fromisoformat(t) for t in access_times
                            ]
            
            except Exception as e:
                print(f"Error loading memories: {e}")
                self.short_term_memory = {}
                self.long_term_memory = {}
                self.access_history = defaultdict(list)

    def save_memories(self) -> None:
        """Save memories to storage."""
        with self.file_lock:
            try:
                temp_file = self.memories_file.with_suffix('.tmp')
                
                data = {
                    "short_term": [
                        self._memory_to_dict(mem)
                        for mem in self.short_term_memory.values()
                    ],
                    "long_term": [
                        self._memory_to_dict(mem)
                        for mem in self.long_term_memory.values()
                    ],
                    "access_history": {
                        mem_id: [t.isoformat() for t in times]
                        for mem_id, times in self.access_history.items()
                    }
                }
                
                with open(temp_file, 'w') as f:
                    json.dump(data, f)
                
                temp_file.replace(self.memories_file)
            
            except Exception as e:
                print(f"Error saving memories: {e}")
                if temp_file.exists():
                    temp_file.unlink()

    def _create_memory_from_dict(self, data: Dict) -> Memory:
        """
        Create a Memory object from dictionary data.
        
        Args:
            data (Dict): Memory data
            
        Returns:
            Memory: Created memory object
        """
        return Memory(
            id=data["id"],
            content=data["content"],
            embedding=data["embedding"],
            token_count=data["token_count"],
            importance=data["importance"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            consolidated=data["consolidated"],
            context=data["context"],
            relationships=set(data["relationships"])
        )

    def _memory_to_dict(self, memory: Memory) -> Dict:
        """
        Convert Memory object to dictionary.
        
        Args:
            memory (Memory): Memory object
            
        Returns:
            Dict: Dictionary representation
        """
        data = asdict(memory)
        data["created_at"] = memory.created_at.isoformat()
        data["last_accessed"] = memory.last_accessed.isoformat()
        data["relationships"] = list(memory.relationships)
        return data

    def add_memory(self, content: str, embedding: List[float],
                  token_count: int, context: Dict = None) -> str:
        """
        Add a new memory to short-term storage.
        
        Args:
            content (str): Memory content
            embedding (List[float]): Content embedding
            token_count (int): Number of tokens
            context (Dict, optional): Additional context
            
        Returns:
            str: Memory ID
        """
        with self.lock:
            try:
                memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                
                memory = Memory(
                    id=memory_id,
                    content=content,
                    embedding=embedding,
                    token_count=token_count,
                    importance=self._calculate_initial_importance(content, context),
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    consolidated=False,
                    context=context or {},
                    relationships=set()
                )
                
                # Add to short-term memory
                self.short_term_memory[memory_id] = memory
                self.access_history[memory_id].append(datetime.now())
                
                # Check capacity
                if len(self.short_term_memory) > self.config["short_term_capacity"]:
                    self._consolidate_lowest_importance()
                
                self.save_memories()
                return memory_id
            
            except Exception as e:
                print(f"Error adding memory: {e}")
                return None

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id (str): Memory ID
            
        Returns:
            Optional[Memory]: Memory object if found
        """
        try:
            memory = (self.short_term_memory.get(memory_id) or 
                     self.long_term_memory.get(memory_id))
            
            if memory:
                self._update_access_stats(memory)
            
            return memory
        
        except Exception as e:
            print(f"Error retrieving memory: {e}")
            return None

    def get_related_memories(self, memory_id: str,
                           limit: int = 10) -> List[Memory]:
        """
        Get memories related to the given memory.
        
        Args:
            memory_id (str): Memory ID
            limit (int): Maximum number of related memories
            
        Returns:
            List[Memory]: Related memories
        """
        try:
            memory = self.get_memory(memory_id)
            if not memory:
                return []
            
            related_memories = []
            for rel_id in memory.relationships:
                rel_memory = self.get_memory(rel_id)
                if rel_memory:
                    related_memories.append(rel_memory)
            
            # Sort by importance and recency
            related_memories.sort(
                key=lambda m: (m.importance, m.last_accessed),
                reverse=True
            )
            
            return related_memories[:limit]
        
        except Exception as e:
            print(f"Error getting related memories: {e}")
            return []

    def get_unconsolidated_memories(self, limit: int = 100) -> List[Memory]:
        """
        Get unconsolidated memories for processing.
        
        Args:
            limit (int): Maximum number of memories to return
            
        Returns:
            List[Memory]: Unconsolidated memories
        """
        try:
            unconsolidated = [
                mem for mem in self.short_term_memory.values()
                if not mem.consolidated
            ]
            
            # Sort by importance and age
            unconsolidated.sort(
                key=lambda m: (m.importance, datetime.now() - m.created_at),
                reverse=True
            )
            
            return unconsolidated[:limit]
        
        except Exception as e:
            print(f"Error getting unconsolidated memories: {e}")
            return []

    def update_memory_importance(self, memory_id: str,
                               importance_delta: float) -> None:
        """
        Update importance score of a memory.
        
        Args:
            memory_id (str): Memory ID
            importance_delta (float): Change in importance
        """
        try:
            memory = self.get_memory(memory_id)
            if memory:
                memory.importance = max(0.0, min(1.0,
                                               memory.importance + importance_delta))
                self.save_memories()
        
        except Exception as e:
            print(f"Error updating memory importance: {e}")

    def consolidate_memory(self, memory_id: str) -> bool:
        """
        Move a memory from short-term to long-term storage and update knowledge graph.
        
        Args:
            memory_id (str): Memory ID
            
        Returns:
            bool: Success status
        """
        with self.lock:
            try:
                memory = self.short_term_memory.get(memory_id)
                if not memory:
                    return False
                
                # Move to long-term memory
                memory.consolidated = True
                self.long_term_memory[memory_id] = memory
                del self.short_term_memory[memory_id]
                
                # Update knowledge graph if available
                if self.knowledge_graph:
                    self.knowledge_graph.add_memory_node(
                        memory_id=memory_id,
                        content=memory.content,
                        embedding=memory.embedding,
                        context=memory.context
                    )
                
                self.save_memories()
                return True
            
            except Exception as e:
                print(f"Error consolidating memory: {e}")
                return False

    def _calculate_initial_importance(self, content: str,
                                    context: Dict = None) -> float:
        """
        Calculate initial importance score for a memory.
        
        Args:
            content (str): Memory content
            context (Dict, optional): Additional context
            
        Returns:
            float: Importance score (0-1)
        """
        importance = 0.5  # Base importance
        
        if context:
            # Adjust based on context
            if context.get("user_marked_important"):
                importance += 0.3
            if context.get("error_related"):
                importance += 0.2
            if context.get("task_critical"):
                importance += 0.2
        
        return min(1.0, importance)

    def _update_access_stats(self, memory: Memory) -> None:
        """
        Update access statistics for a memory.
        
        Args:
            memory (Memory): Memory to update
        """
        try:
            current_time = datetime.now()
            
            # Update memory stats
            memory.last_accessed = current_time
            memory.access_count += 1
            
            # Update access history
            self.access_history[memory.id].append(current_time)
            
            # Cleanup old access history
            cutoff = current_time - timedelta(days=self.config["access_window"])
            self.access_history[memory.id] = [
                t for t in self.access_history[memory.id]
                if t > cutoff
            ]
            
            # Update importance based on access patterns
            access_frequency = len(self.access_history[memory.id])
            importance_boost = min(0.1, access_frequency * 0.01)
            memory.importance = min(1.0, memory.importance + importance_boost)
            
            self.save_memories()
        
        except Exception as e:
            print(f"Error updating access stats: {e}")

    def _consolidate_lowest_importance(self) -> None:
        """Consolidate least important memories to maintain capacity."""
        try:
            if not self.short_term_memory:
                return
            
            # Get current system state from usage tracker
            system_state = {}
            if self.usage_tracker:
                system_state = self.usage_tracker.get_current_state()
            
            # Adjust consolidation threshold based on system state
            threshold = self.config["consolidation_threshold"]
            if system_state.get("memory_pressure", 0) > 0.8:
                threshold *= 1.2  # Be more aggressive
            elif system_state.get("idle_time", 0) > 300:  # 5 minutes
                threshold *= 0.8  # Be less aggressive
            
            # Find memories below threshold
            to_consolidate = [
                mem for mem in self.short_term_memory.values()
                if mem.importance < threshold
            ]
            
            # Sort by importance (ascending)
            to_consolidate.sort(key=lambda m: m.importance)
            
            # Consolidate until we're under capacity
            while (len(self.short_term_memory) > self.config["short_term_capacity"]
                   and to_consolidate):
                memory = to_consolidate.pop(0)
                self.consolidate_memory(memory.id)
                
                # Update usage tracker if available
                if self.usage_tracker:
                    self.usage_tracker.log_memory_consolidation(
                        memory_id=memory.id,
                        importance=memory.importance,
                        token_count=memory.token_count
                    )
        
        except Exception as e:
            print(f"Error consolidating memories: {e}")

    def cleanup_old_memories(self) -> int:
        """
        Remove old, unimportant memories.
        
        Returns:
            int: Number of memories cleaned up
        """
        with self.lock:
            try:
                cleaned_count = 0
                current_time = datetime.now()
                cutoff = current_time - timedelta(days=self.config["access_window"])
                
                # Clean short-term memory
                to_remove = []
                for mem_id, memory in self.short_term_memory.items():
                    if (memory.last_accessed < cutoff and
                            memory.importance < self.config["importance_threshold"]):
                        to_remove.append(mem_id)
                
                for mem_id in to_remove:
                    del self.short_term_memory[mem_id]
                    cleaned_count += 1
                
                # Clean long-term memory
                to_remove = []
                for mem_id, memory in self.long_term_memory.items():
                    if (memory.last_accessed < cutoff and
                            memory.importance < self.config["importance_threshold"]):
                        to_remove.append(mem_id)
                
                for mem_id in to_remove:
                    del self.long_term_memory[mem_id]
                    cleaned_count += 1
                
                if cleaned_count > 0:
                    self.save_memories()
                
                return cleaned_count
            
            except Exception as e:
                print(f"Error cleaning up memories: {e}")
                return 0

    def get_memory_pressure(self) -> float:
        """
        Calculate current memory pressure (0-1) considering system state.
        
        Returns:
            float: Memory pressure score
        """
        try:
            # Base pressure calculation
            short_term_pressure = (len(self.short_term_memory) /
                                 self.config["short_term_capacity"])
            long_term_pressure = (len(self.long_term_memory) /
                                self.config["long_term_capacity"])
            
            # Get additional context from usage tracker
            usage_context = {}
            if self.usage_tracker:
                usage_context = self.usage_tracker.get_memory_context()
            
            # Adjust pressure based on usage patterns
            pressure_adjustment = 0.0
            if usage_context.get("recent_usage", 0) > 1000:  # High recent usage
                pressure_adjustment += 0.1
            if usage_context.get("error_rate", 0) > 0.1:  # High error rate
                pressure_adjustment += 0.2
            
            # Weight short-term more heavily as it's more critical
            base_pressure = (0.7 * short_term_pressure) + (0.3 * long_term_pressure)
            return min(1.0, base_pressure + pressure_adjustment)
        
        except Exception as e:
            print(f"Error calculating memory pressure: {e}")
            return 0.0

    def get_access_frequency(self, memory_id: str,
                           window_days: int = None) -> float:
        """
        Get access frequency for a memory.
        
        Args:
            memory_id (str): Memory ID
            window_days (int, optional): Time window to consider
            
        Returns:
            float: Access frequency (accesses per day)
        """
        try:
            if not window_days:
                window_days = self.config["access_window"]
            
            cutoff = datetime.now() - timedelta(days=window_days)
            recent_accesses = [
                t for t in self.access_history.get(memory_id, [])
                if t > cutoff
            ]
            
            if not recent_accesses:
                return 0.0
            
            # Calculate accesses per day
            days_span = (max(recent_accesses) - min(recent_accesses)).days + 1
            return len(recent_accesses) / max(1, days_span)
        
        except Exception as e:
            print(f"Error calculating access frequency: {e}")
            return 0.0

    def find_similar_memories(self, embedding: List[float],
                            threshold: float = 0.7,
                            limit: int = 10) -> List[Memory]:
        """
        Find memories with similar embeddings.
        
        Args:
            embedding (List[float]): Reference embedding
            threshold (float): Similarity threshold
            limit (int): Maximum number of results
            
        Returns:
            List[Memory]: Similar memories
        """
        try:
            similar_memories = []
            
            # Check both memory stores
            for memory in list(self.short_term_memory.values()) + list(self.long_term_memory.values()):
                similarity = self._calculate_similarity(embedding, memory.embedding)
                if similarity >= threshold:
                    similar_memories.append((similarity, memory))
            
            # Sort by similarity
            similar_memories.sort(reverse=True, key=lambda x: x[0])
            return [mem for _, mem in similar_memories[:limit]]
        
        except Exception as e:
            print(f"Error finding similar memories: {e}")
            return []

    def _calculate_similarity(self, embedding1: List[float],
                            embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1 (List[float]): First embedding
            embedding2 (List[float]): Second embedding
            
        Returns:
            float: Similarity score
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def update_relationships(self, source_id: str,
                           target_ids: List[str],
                           strengths: List[float]) -> None:
        """
        Update relationship strengths between memories.
        
        Args:
            source_id (str): Source memory ID
            target_ids (List[str]): Target memory IDs
            strengths (List[float]): Relationship strengths
        """
        try:
            source_memory = self.get_memory(source_id)
            if not source_memory:
                return
            
            # Update relationships
            current_relationships = source_memory.relationships
            new_relationships = set()
            
            for target_id, strength in zip(target_ids, strengths):
                if strength >= self.config["consolidation_threshold"]:
                    new_relationships.add(target_id)
            
            # Keep strongest relationships within limit
            if len(new_relationships) > self.config["max_relationships"]:
                new_relationships = set(sorted(
                    new_relationships,
                    key=lambda x: strengths[target_ids.index(x)],
                    reverse=True
                )[:self.config["max_relationships"]])
            
            source_memory.relationships = new_relationships
            self.save_memories()
        
        except Exception as e:
            print(f"Error updating relationships: {e}")

    def get_memory_stats(self) -> Dict:
        """
        Get current memory system statistics.
        
        Returns:
            Dict: Memory statistics
        """
        try:
            return {
                "short_term_count": len(self.short_term_memory),
                "long_term_count": len(self.long_term_memory),
                "short_term_capacity": self.config["short_term_capacity"],
                "long_term_capacity": self.config["long_term_capacity"],
                "memory_pressure": self.get_memory_pressure(),
                "unconsolidated_count": len([
                    m for m in self.short_term_memory.values()
                    if not m.consolidated
                ]),
                "average_importance": np.mean([
                    m.importance
                    for m in list(self.short_term_memory.values()) +
                            list(self.long_term_memory.values())
                ]) if self.short_term_memory or self.long_term_memory else 0.0
            }
        
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}
