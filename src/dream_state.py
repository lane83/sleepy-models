#!/usr/bin/env python3
"""
Dream state system for Sleepy-Models.
Handles memory consolidation, knowledge graph updates, and system optimization.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import json
import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class DreamStateMetrics:
    """Metrics collected during dream state."""
    start_time: datetime
    end_time: Optional[datetime]
    tokens_processed: int
    memories_consolidated: int
    knowledge_connections_made: int
    optimization_score: float


class DreamState:
    """Manages the dream state process and memory consolidation."""
    
    def __init__(self,
                 knowledge_graph,
                 memory_manager,
                 data_dir: str = "data"):
        """
        Initialize the dream state system.
        
        Args:
            knowledge_graph: Knowledge graph system
            memory_manager: Memory management system
            data_dir (str): Directory to store dream state data
        """
        self.knowledge_graph = knowledge_graph
        self.memory_manager = memory_manager
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dream_state_file = self.data_dir / "dream_states.json"
        self.active_dream = False
        self.dream_thread = None
        self.dream_states: List[DreamStateMetrics] = []
        
        # Load configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load dream state configuration from file or use defaults."""
        config_file = self.data_dir / "dream_config.json"
        default_config = {
            "min_dream_duration": 60,  # seconds
            "max_dream_duration": 300,  # seconds
            "memory_batch_size": 100,
            "consolidation_threshold": 0.7,
            "connection_strength_threshold": 0.5,
            "parallel_workers": 4,
            "memory_consolidation_batch_size": 25,
            "knowledge_optimization_batch_size": 50,
            "system_optimization_weights": {
                "memory_cleanup": 0.3,
                "cache_optimization": 0.2,
                "model_optimization": 0.5
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for key in user_config:
                        if key in default_config:
                            default_config[key] = user_config[key]
            except Exception as e:
                print(f"Error loading dream config: {e}")
        
        return default_config
        
        self.load_dream_states()

    def load_dream_states(self) -> None:
        """Load previous dream state records."""
        try:
            if self.dream_state_file.exists():
                with open(self.dream_state_file, 'r') as f:
                    data = json.load(f)
                    self.dream_states = [
                        DreamStateMetrics(
                            start_time=datetime.fromisoformat(d['start_time']),
                            end_time=datetime.fromisoformat(d['end_time']) if d['end_time'] else None,
                            tokens_processed=d['tokens_processed'],
                            memories_consolidated=d['memories_consolidated'],
                            knowledge_connections_made=d['knowledge_connections_made'],
                            optimization_score=d['optimization_score']
                        )
                        for d in data
                    ]
        except Exception as e:
            print(f"Error loading dream states: {e}")
            self.dream_states = []

    def save_dream_states(self) -> None:
        """Save dream state records."""
        try:
            data = [
                {
                    'start_time': d.start_time.isoformat(),
                    'end_time': d.end_time.isoformat() if d.end_time else None,
                    'tokens_processed': d.tokens_processed,
                    'memories_consolidated': d.memories_consolidated,
                    'knowledge_connections_made': d.knowledge_connections_made,
                    'optimization_score': d.optimization_score
                }
                for d in self.dream_states
            ]
            
            # Atomic write
            temp_file = self.dream_state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f)
            temp_file.replace(self.dream_state_file)
        
        except Exception as e:
            print(f"Error saving dream states: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def check_dream_needed(self, model_manager) -> Tuple[bool, List[str]]:
        """
        Check if a dream state is needed based on system metrics.
        
        Args:
            model_manager: Model management system
            
        Returns:
            Tuple[bool, List[str]]: (need_dream, reasons)
        """
        reasons = []
        need_dream = False
        
        # Get adaptive thresholds
        thresholds = self._get_adaptive_thresholds(model_manager)
        
        # Check token usage
        total_tokens = model_manager.get_total_tokens()
        token_threshold = thresholds.get("token_usage", 0.8)
        if total_tokens > model_manager.get_token_limit() * token_threshold:
            reasons.append(f"High token usage ({total_tokens} > {token_threshold*100:.0f}%)")
            need_dream = True
        
        # Check memory pressure
        memory_pressure = self.memory_manager.get_memory_pressure()
        memory_threshold = thresholds.get("memory_pressure", 0.85)
        if memory_pressure > memory_threshold:
            reasons.append(f"High memory pressure ({memory_pressure:.0%} > {memory_threshold:.0%})")
            need_dream = True
        
        # Check error rates
        error_rate = model_manager.get_error_rate()
        error_threshold = thresholds.get("error_rate", 0.1)
        if error_rate > error_threshold:
            reasons.append(f"Elevated error rate ({error_rate:.0%} > {error_threshold:.0%})")
            need_dream = True
        
        # Check time since last dream
        last_dream = self.get_last_dream_time()
        if last_dream:
            time_since_dream = datetime.now() - last_dream
            dream_interval = thresholds.get("dream_interval", timedelta(hours=1))
            if time_since_dream > dream_interval:
                reasons.append(f"Time since last dream ({time_since_dream} > {dream_interval})")
                need_dream = True
        
        return need_dream, reasons

    def _get_adaptive_thresholds(self, model_manager) -> Dict[str, Any]:
        """
        Calculate adaptive thresholds based on comprehensive system metrics.
        
        Args:
            model_manager: Model management system
            
        Returns:
            Dict[str, Any]: Adaptive threshold values with dynamic weighting
        """
        thresholds = {
            "token_usage": 0.8,
            "memory_pressure": 0.85,
            "error_rate": 0.1,
            "dream_interval": timedelta(hours=1),
            "weights": {
                "token_usage": 0.35,
                "memory_pressure": 0.3,
                "error_rate": 0.2,
                "dream_interval": 0.15
            }
        }
        
        try:
            # Enhanced token usage analysis with exponential smoothing
            total_tokens = model_manager.get_total_tokens()
            token_limit = model_manager.get_token_limit()
            
            # Calculate token usage ratio with dynamic scaling
            token_usage_ratio = total_tokens / max(token_limit, 1)
            
            # Apply exponential decay for recent usage emphasis
            decay_factor = 0.7
            recent_usage = token_usage_ratio * decay_factor
            avg_usage = recent_usage + (1 - decay_factor) * thresholds["token_usage"]
            
            # Adjust based on usage volatility
            usage_variability = abs(token_usage_ratio - thresholds["token_usage"])
            if usage_variability > 0.2:
                avg_usage *= 1.1  # Increase sensitivity for volatile usage
            
            thresholds["token_usage"] = min(0.95, max(0.6, avg_usage))
        except Exception as e:
            print(f"Error analyzing token usage patterns: {e}")
        
        try:
            # Memory pressure analysis with trend detection
            memory_stats = self.memory_manager.get_memory_usage_stats()
            if memory_stats:
                # Detect upward/downward trends
                trend = memory_stats.get("trend", 0)  # -1 = decreasing, 0 = stable, 1 = increasing
                avg_pressure = memory_stats.get("average_pressure", 0.85)
                
                # Adjust based on trend direction
                if trend > 0:
                    avg_pressure *= 1.1  # Be more aggressive with increasing pressure
                elif trend < 0:
                    avg_pressure *= 0.9  # Be more lenient with decreasing pressure
                
                # Incorporate memory type distribution
                memory_distribution = self.memory_manager.get_memory_distribution()
                if memory_distribution.get("volatile_ratio", 0) > 0.3:
                    avg_pressure *= 0.95  # Adjust for volatile memory
                
                thresholds["memory_pressure"] = min(0.97, max(0.7, avg_pressure))
        except Exception as e:
            print(f"Error analyzing memory pressure patterns: {e}")
        
        try:
            # Enhanced error rate analysis with severity weighting
            error_rate = model_manager.get_error_rate()
            
            # Calculate weighted error rate with exponential smoothing
            decay_factor = 0.8
            recent_error_rate = error_rate * decay_factor
            avg_error_rate = recent_error_rate + (1 - decay_factor) * thresholds["error_rate"]
            
            # Adjust based on error severity
            error_stats = model_manager.get_error_rate_stats()
            if error_stats:
                severity_weights = {
                    "critical": 1.5,
                    "high": 1.2,
                    "medium": 1.0,
                    "low": 0.8
                }
                weighted_errors = sum(
                    error_stats.get(f"{level}_errors", 0) * weight
                    for level, weight in severity_weights.items()
                )
                total_errors = sum(
                    error_stats.get(f"{level}_errors", 0)
                    for level in severity_weights
                )
                
                if total_errors > 0:
                    severity_factor = weighted_errors / total_errors
                    avg_error_rate *= severity_factor
            
            thresholds["error_rate"] = min(0.2, max(0.03, avg_error_rate))
        except Exception as e:
            print(f"Error analyzing error rate patterns: {e}")
        
        try:
            # Enhanced dream interval analysis with workload awareness
            dream_stats = self.get_dream_stats(days=7)
            if dream_stats and dream_stats["total_dreams"] > 0:
                # Calculate base interval with exponential smoothing
                avg_interval = timedelta(
                    seconds=dream_stats["total_dreams"] / 
                    (time.time() - self.dream_states[0].start_time.timestamp())
                )
                
                # Adjust based on current workload and cache performance
                workload_factor = model_manager.get_current_workload()
                cache_hit_rate = sum(
                    stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1)
                    for stats in model_manager.model_stats.values()
                ) / max(len(model_manager.model_stats), 1)
                
                # More frequent dreams under heavy load or low cache performance
                if workload_factor > 1.5 or cache_hit_rate < 0.3:
                    avg_interval *= 0.8
                elif workload_factor < 0.5 and cache_hit_rate > 0.7:
                    avg_interval *= 1.2
                
                thresholds["dream_interval"] = min(
                    timedelta(hours=3),
                    max(timedelta(minutes=15), avg_interval)
                )
        except Exception as e:
            print(f"Error analyzing dream interval patterns: {e}")
            thresholds["dream_interval"] = timedelta(hours=1)
        
        # Calculate dynamic weights based on system state
        try:
            system_state = model_manager.get_system_state()
            if system_state:
                # Adjust weights based on criticality
                if system_state.get("critical_errors", 0) > 0:
                    thresholds["weights"]["error_rate"] = 0.4
                    thresholds["weights"]["token_usage"] = 0.3
                elif system_state.get("memory_warnings", 0) > 0:
                    thresholds["weights"]["memory_pressure"] = 0.4
                    thresholds["weights"]["dream_interval"] = 0.2
                
                # Incorporate cache performance into weights
                cache_hit_rate = sum(
                    stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1)
                    for stats in model_manager.model_stats.values()
                ) / max(len(model_manager.model_stats), 1)
                
                if cache_hit_rate < 0.3:
                    thresholds["weights"]["token_usage"] *= 1.2
                    thresholds["weights"]["memory_pressure"] *= 0.8
                
                # Normalize weights to sum to 1
                total_weight = sum(thresholds["weights"].values())
                for key in thresholds["weights"]:
                    thresholds["weights"][key] /= total_weight
        except Exception as e:
            print(f"Error calculating dynamic weights: {e}")
        
        return thresholds

    def enter_dream_state(self, model_manager, force: bool = False) -> DreamStateMetrics:
        """
        Enter dream state for memory consolidation and optimization.
        
        Args:
            model_manager: Model management system
            force (bool): Force entry into dream state
            
        Returns:
            DreamStateMetrics: Metrics from the dream state
        """
        if self.active_dream:
            raise RuntimeError("Already in dream state")
        
        need_dream, reasons = self.check_dream_needed(model_manager)
        if not (need_dream or force):
            return None
        
        try:
            self.active_dream = True
            current_dream = DreamStateMetrics(
                start_time=datetime.now(),
                end_time=None,
                tokens_processed=0,
                memories_consolidated=0,
                knowledge_connections_made=0,
                optimization_score=0.0
            )
            
            self.dream_states.append(current_dream)
            
            # Start dream state processing in background
            self.dream_thread = threading.Thread(
                target=self._dream_process,
                args=(current_dream, model_manager)
            )
            self.dream_thread.start()
            
            return current_dream
        
        except Exception as e:
            self.active_dream = False
            print(f"Error entering dream state: {e}")
            return None

    def _dream_process(self, metrics: DreamStateMetrics, model_manager) -> None:
        """
        Main dream state processing function.
        
        Args:
            metrics (DreamStateMetrics): Metrics object to update
            model_manager: Model management system
        """
        try:
            start_time = time.time()
            
            # Phase 1: Memory Consolidation (parallel processing)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            memories = self.memory_manager.get_unconsolidated_memories(
                self.config["memory_batch_size"]
            )
            
            # Process memories in batches
            batch_size = self.config["memory_consolidation_batch_size"]
            with ThreadPoolExecutor(max_workers=self.config["parallel_workers"]) as executor:
                for i in range(0, len(memories), batch_size):
                    if self._should_end_dream(start_time):
                        break
                    
                    batch = memories[i:i + batch_size]
                    futures = [
                        executor.submit(self._consolidate_memory, memory, model_manager)
                        for memory in batch
                    ]
                    
                    for future in as_completed(futures):
                        if future.result():
                            metrics.memories_consolidated += 1
                            metrics.tokens_processed += future.result().token_count
            
            # Phase 2: Knowledge Graph Optimization
            if not self._should_end_dream(start_time):
                connections = self._optimize_knowledge_graph()
                metrics.knowledge_connections_made += connections
            
            # Phase 3: System Optimization
            if not self._should_end_dream(start_time):
                metrics.optimization_score = self._optimize_system(model_manager)
            
            # Finalize
            metrics.end_time = datetime.now()
            self.save_dream_states()
        
        except Exception as e:
            print(f"Error in dream process: {e}")
        
        finally:
            self.active_dream = False

    def _should_end_dream(self, start_time: float) -> bool:
        """
        Check if dream state should end.
        
        Args:
            start_time (float): Start time of dream state
            
        Returns:
            bool: Whether dream state should end
        """
        elapsed = time.time() - start_time
        return elapsed >= self.config["max_dream_duration"]

    def _consolidate_memory(self, memory: Dict, model_manager) -> bool:
        """
        Consolidate a single memory.
        
        Args:
            memory (Dict): Memory to consolidate
            model_manager: Model management system
            
        Returns:
            bool: Success status
        """
        try:
            # Extract key information
            embedding = model_manager.get_embedding(memory.content)
            
            # Find related memories
            related = self.knowledge_graph.find_related(
                embedding,
                threshold=self.config["connection_strength_threshold"]
            )
            
            # Create new connections
            for rel_memory in related:
                self.knowledge_graph.add_connection(
                    memory.id,
                    rel_memory.id,
                    embedding.similarity(rel_memory.embedding)
                )
            
            # Update memory state
            memory.consolidated = True
            memory.last_accessed = datetime.now()
            
            return True
        
        except Exception as e:
            print(f"Error consolidating memory: {e}")
            return False

    def _optimize_knowledge_graph(self) -> int:
        """
        Optimize knowledge graph connections with advanced analysis.
        
        Returns:
            int: Number of connections optimized
        """
        connections_modified = 0
        
        try:
            # Identify weak connections in batches
            batch_size = self.config["knowledge_optimization_batch_size"]
            
            while True:
                weak_connections = self.knowledge_graph.find_weak_connections(
                    threshold=self.config["connection_strength_threshold"],
                    limit=batch_size
                )
                
                if not weak_connections:
                    break
                
                # Analyze connection patterns
                connection_groups = self._analyze_connection_patterns(weak_connections)
                
                # Process connections based on analysis
                for group in connection_groups:
                    if len(group) > 1:
                        # Merge similar connections
                        self.knowledge_graph.merge_connections(group)
                        connections_modified += len(group)
                    else:
                        # Strengthen or prune individual connections
                        conn = group[0]
                        if self._should_strengthen_connection(conn):
                            self.knowledge_graph.strengthen_connection(conn)
                            connections_modified += 1
                        else:
                            self.knowledge_graph.remove_connection(conn)
                            connections_modified += 1
            
        except Exception as e:
            print(f"Error optimizing knowledge graph: {e}")
            # Continue with partial results if optimization fails
            pass
            
        return connections_modified
        
    def _analyze_connection_patterns(self, connections: List) -> List[List]:
        """
        Analyze connection patterns to identify groups of related connections.
        
        Args:
            connections (List): List of connections to analyze
            
        Returns:
            List[List]: Groups of related connections
        """
        if not connections:
            return []

        try:
            # Create similarity matrix
            embeddings = [
                self.knowledge_graph.get_connection_embedding(conn)
                for conn in connections
            ]
            
            # Cluster similar connections
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings)
            
            # Group connections by cluster
            groups = []
            for label in set(clustering.labels_):
                if label != -1:  # Ignore noise
                    group = [
                        conn for conn, lbl in zip(connections, clustering.labels_)
                        if lbl == label
                    ]
                    groups.append(group)
            
            # Add ungrouped connections
            for conn, label in zip(connections, clustering.labels_):
                if label == -1:
                    groups.append([conn])
            
            return groups
        
        except ImportError as e:
            print(f"Required clustering library not available: {e}")
            # Fallback to simple grouping by direct similarity
            return [[conn] for conn in connections]
        except Exception as e:
            print(f"Error analyzing connection patterns: {e}")
            return [[conn] for conn in connections]

    def _should_strengthen_connection(self, connection: Dict) -> bool:
        """
        Determine if a connection should be strengthened.
        
        Args:
            connection (Dict): Connection to evaluate
            
        Returns:
            bool: Whether to strengthen the connection
        """
        # Check access frequency
        access_frequency = self.memory_manager.get_access_frequency(
            connection.source_id,
            connection.target_id
        )
        
        # Check semantic similarity
        similarity = self.knowledge_graph.get_connection_strength(connection)
        
        return (access_frequency > 0.3 and 
                similarity > self.config["connection_strength_threshold"])

    def _optimize_system(self, model_manager) -> float:
        """
        Perform comprehensive system-level optimizations.
        
        Args:
            model_manager: Model management system
            
        Returns:
            float: Optimization score (0-1)
        """
        try:
            improvements = []
            weights = self.config["system_optimization_weights"]
            
            # Memory cleanup
            freed_memory = self.memory_manager.cleanup_old_memories()
            if freed_memory > 0:
                improvements.append(weights["memory_cleanup"])
            
            # Cache optimization
            cache_improved = model_manager.optimize_cache()
            if cache_improved:
                improvements.append(weights["cache_optimization"])
            
            # Model optimization
            model_improved = model_manager.optimize_models()
            if model_improved:
                improvements.append(weights["model_optimization"])
            
            # Resource allocation optimization
            resource_improved = model_manager.optimize_resource_allocation()
            if resource_improved:
                improvements.append(0.2)  # Additional weight for resource optimization
            
            # Calculate weighted score
            total_weight = sum(weights.values()) + 0.2  # Include resource optimization weight
            return sum(improvements) / total_weight if improvements else 0.0
        
        except Exception as e:
            print(f"Error optimizing system: {e}")
            return 0.0

    def get_last_dream_time(self) -> Optional[datetime]:
        """
        Get the time of the last completed dream state.
        
        Returns:
            Optional[datetime]: Time of last dream state or None
        """
        completed_dreams = [d for d in self.dream_states if d.end_time]
        if not completed_dreams:
            return None
        return max(d.end_time for d in completed_dreams)

    def get_dream_stats(self, days: int = 7) -> Dict:
        """
        Get dream state statistics for the specified period.
        
        Args:
            days (int): Number of days to include
            
        Returns:
            Dict: Dream state statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_dreams = [
            d for d in self.dream_states
            if d.start_time > cutoff and d.end_time
        ]
        
        if not recent_dreams:
            return {
                "total_dreams": 0,
                "average_duration": 0,
                "total_memories_consolidated": 0,
                "total_connections_made": 0,
                "average_optimization_score": 0
            }
        
        durations = [
            (d.end_time - d.start_time).total_seconds()
            for d in recent_dreams
        ]
        
        return {
            "total_dreams": len(recent_dreams),
            "average_duration": np.mean(durations),
            "total_memories_consolidated": sum(d.memories_consolidated for d in recent_dreams),
            "total_connections_made": sum(d.knowledge_connections_made for d in recent_dreams),
            "average_optimization_score": np.mean([d.optimization_score for d in recent_dreams])
        }
