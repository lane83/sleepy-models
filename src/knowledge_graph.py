#!/usr/bin/env python3
"""
Knowledge graph system for Sleepy-Models.
Manages relationships between memories and concepts.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import networkx as nx
from filelock import FileLock
import threading


@dataclass
class Node:
    """Data class for knowledge graph nodes."""
    id: str
    type: str  # 'memory', 'concept', or 'relationship'
    content: str
    embedding: List[float]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    importance: float
    metadata: Dict


@dataclass
class Edge:
    """Data class for knowledge graph edges."""
    source_id: str
    target_id: str
    type: str
    strength: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    metadata: Dict


class KnowledgeGraph:
    """Manages the knowledge graph structure and operations."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the knowledge graph.
        
        Args:
            data_dir (str): Directory to store graph data
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.graph_file = self.data_dir / "knowledge_graph.json"
        self.lock_file = self.data_dir / "graph.lock"
        self.file_lock = FileLock(str(self.lock_file))
        
        # Initialize NetworkX graph
        self.graph = nx.DiGraph()
        
        # Cache for embeddings and computations
        self.embedding_cache: Dict[str, List[float]] = {}
        self.similarity_cache: Dict[str, Dict[str, float]] = {}
        
        # Configuration
        self.config = {
            "min_edge_strength": 0.3,
            "max_node_connections": 100,
            "similarity_threshold": 0.7,
            "importance_decay_rate": 0.1,  # per day
            "cache_ttl": 3600,  # seconds
            "max_path_length": 5
        }
        
        self.lock = threading.Lock()
        self.load_graph()

    def load_graph(self) -> None:
        """Load graph data from storage."""
        with self.file_lock:
            try:
                if self.graph_file.exists():
                    with open(self.graph_file, 'r') as f:
                        data = json.load(f)
                        
                        # Load nodes
                        for node_data in data.get("nodes", []):
                            node = self._create_node_from_dict(node_data)
                            self.graph.add_node(
                                node.id,
                                node_data=node
                            )
                        
                        # Load edges
                        for edge_data in data.get("edges", []):
                            edge = self._create_edge_from_dict(edge_data)
                            self.graph.add_edge(
                                edge.source_id,
                                edge.target_id,
                                edge_data=edge
                            )
            
            except Exception as e:
                self.logger.error(f"Error loading knowledge graph: {e}", exc_info=True)
                self.graph = nx.DiGraph()

    def save_graph(self) -> None:
        """Save graph data to storage."""
        with self.file_lock:
            try:
                temp_file = self.graph_file.with_suffix('.tmp')
                
                data = {
                    "nodes": [
                        self._node_to_dict(node_data)
                        for _, node_data in self.graph.nodes(data="node_data")
                    ],
                    "edges": [
                        self._edge_to_dict(edge_data)
                        for _, _, edge_data in self.graph.edges(data="edge_data")
                    ]
                }
                
                with open(temp_file, 'w') as f:
                    json.dump(data, f)
                
                temp_file.replace(self.graph_file)
            
            except Exception as e:
                self.logger.error(f"Error saving knowledge graph: {e}", exc_info=True)
                if temp_file.exists():
                    temp_file.unlink()

    def _create_node_from_dict(self, data: Dict) -> Node:
        """
        Create a Node object from dictionary data.
        
        Args:
            data (Dict): Node data
            
        Returns:
            Node: Created node object
        """
        return Node(
            id=data["id"],
            type=data["type"],
            content=data["content"],
            embedding=data["embedding"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            importance=data["importance"],
            metadata=data["metadata"]
        )

    def _create_edge_from_dict(self, data: Dict) -> Edge:
        """
        Create an Edge object from dictionary data.
        
        Args:
            data (Dict): Edge data
            
        Returns:
            Edge: Created edge object
        """
        return Edge(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            strength=data["strength"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            metadata=data["metadata"]
        )

    def _node_to_dict(self, node: Node) -> Dict:
        """
        Convert Node object to dictionary.
        
        Args:
            node (Node): Node object
            
        Returns:
            Dict: Dictionary representation
        """
        data = asdict(node)
        data["created_at"] = node.created_at.isoformat()
        data["last_accessed"] = node.last_accessed.isoformat()
        return data

    def _edge_to_dict(self, edge: Edge) -> Dict:
        """
        Convert Edge object to dictionary.
        
        Args:
            edge (Edge): Edge object
            
        Returns:
            Dict: Dictionary representation
        """
        data = asdict(edge)
        data["created_at"] = edge.created_at.isoformat()
        data["last_accessed"] = edge.last_accessed.isoformat()
        return data

    def add_node(self, content: str, node_type: str,
                embedding: List[float], metadata: Dict = None) -> str:
        """
        Add a new node to the graph.
        
        Args:
            content (str): Node content
            node_type (str): Type of node
            embedding (List[float]): Content embedding
            metadata (Dict, optional): Additional metadata
            
        Returns:
            str: Node ID
        """
        with self.lock:
            try:
                node_id = f"{node_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                
                node = Node(
                    id=node_id,
                    type=node_type,
                    content=content,
                    embedding=embedding,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    importance=self._calculate_initial_importance(content, metadata),
                    metadata=metadata or {}
                )
                
                self.graph.add_node(node_id, node_data=node)
                self.embedding_cache[node_id] = embedding
                
                # Find and create connections to similar nodes
                self._create_initial_connections(node_id, embedding)
                
                self.save_graph()
                return node_id
            
            except Exception as e:
                self.logger.error(f"Error adding node: {e}", exc_info=True)
                return None

    def add_edge(self, source_id: str, target_id: str,
                edge_type: str, strength: float,
                metadata: Dict = None) -> bool:
        """
        Add a new edge between nodes.
        
        Args:
            source_id (str): Source node ID
            target_id (str): Target node ID
            edge_type (str): Type of edge
            strength (float): Edge strength
            metadata (Dict, optional): Additional metadata
            
        Returns:
            bool: Success status
        """
        with self.lock:
            try:
                if not (self.graph.has_node(source_id) and
                       self.graph.has_node(target_id)):
                    return False
                
                edge = Edge(
                    source_id=source_id,
                    target_id=target_id,
                    type=edge_type,
                    strength=strength,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    metadata=metadata or {}
                )
                
                self.graph.add_edge(
                    source_id,
                    target_id,
                    edge_data=edge
                )
                
                # Update similarity cache
                self.similarity_cache.setdefault(source_id, {})[target_id] = strength
                
                self.save_graph()
                return True
            
            except Exception as e:
                self.logger.error(f"Error adding edge: {e}", exc_info=True)
                return False

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get a node by ID.
        
        Args:
            node_id (str): Node ID
            
        Returns:
            Optional[Node]: Node object if found
        """
        try:
            if self.graph.has_node(node_id):
                node_data = self.graph.nodes[node_id]["node_data"]
                self._update_node_access(node_id)
                return node_data
            return None
        
        except Exception as e:
                self.logger.error(f"Error getting node: {e}", exc_info=True)
                return None

    def get_edge(self, source_id: str, target_id: str) -> Optional[Edge]:
        """
        Get an edge between nodes.
        
        Args:
            source_id (str): Source node ID
            target_id (str): Target node ID
            
        Returns:
            Optional[Edge]: Edge object if found
        """
        try:
            if self.graph.has_edge(source_id, target_id):
                edge_data = self.graph[source_id][target_id]["edge_data"]
                self._update_edge_access(source_id, target_id)
                return edge_data
            return None
        
        except Exception as e:
                self.logger.error(f"Error getting edge: {e}", exc_info=True)
                return None

    def update_node_importance(self, node_id: str,
                             importance_delta: float) -> None:
        """
        Update node importance score.
        
        Args:
            node_id (str): Node ID
            importance_delta (float): Change in importance
        """
        try:
            if self.graph.has_node(node_id):
                node = self.graph.nodes[node_id]["node_data"]
                node.importance = max(0.0, min(1.0,
                                             node.importance + importance_delta))
                self.save_graph()
        
        except Exception as e:
                self.logger.error(f"Error updating node importance: {e}", exc_info=True)

    def update_edge_strength(self, source_id: str,
                           target_id: str,
                           strength_delta: float) -> None:
        """
        Update edge strength.
        
        Args:
            source_id (str): Source node ID
            target_id (str): Target node ID
            strength_delta (float): Change in strength
        """
        try:
            if self.graph.has_edge(source_id, target_id):
                edge = self.graph[source_id][target_id]["edge_data"]
                edge.strength = max(0.0, min(1.0,
                                           edge.strength + strength_delta))
                
                # Update cache
                self.similarity_cache.setdefault(source_id, {})[target_id] = edge.strength
                
                self.save_graph()
        
        except Exception as e:
                self.logger.error(f"Error updating edge strength: {e}", exc_info=True)

    def find_related(self, embedding: List[float],
                    threshold: float = None,
                    limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find nodes with similar embeddings.
        
        Args:
            embedding (List[float]): Reference embedding
            threshold (float, optional): Similarity threshold
            limit (int): Maximum number of results
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, similarity) pairs
        """
        try:
            if threshold is None:
                threshold = self.config["similarity_threshold"]
            
            similarities = []
            for node_id, node_data in self.graph.nodes(data="node_data"):
                similarity = self._calculate_similarity(
                    embedding,
                    node_data.embedding
                )
                if similarity >= threshold:
                    similarities.append((node_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]
        
        except Exception as e:
                self.logger.error(f"Error finding related nodes: {e}", exc_info=True)
                return []

    def find_path(self, source_id: str,
                 target_id: str,
                 max_length: int = None) -> Optional[List[str]]:
        """
        Find shortest path between nodes.
        
        Args:
            source_id (str): Source node ID
            target_id (str): Target node ID
            max_length (int, optional): Maximum path length
            
        Returns:
            Optional[List[str]]: Path of node IDs if found
        """
        try:
            if max_length is None:
                max_length = self.config["max_path_length"]
            
            if not (self.graph.has_node(source_id) and
                   self.graph.has_node(target_id)):
                return None
            
            try:
                path = nx.shortest_path(
                    self.graph,
                    source=source_id,
                    target=target_id,
                    weight="weight"
                )
                
                if len(path) <= max_length:
                    return path
                return None
            
            except nx.NetworkXNoPath:
                return None
        
        except Exception as e:
                self.logger.error(f"Error finding path: {e}", exc_info=True)
                return None

    def find_weak_connections(self,
                            threshold: float = None) -> List[Tuple[str, str]]:
        """
        Find weak edges in the graph.
        
        Args:
            threshold (float, optional): Strength threshold
            
        Returns:
            List[Tuple[str, str]]: List of weak edges
        """
        try:
            if threshold is None:
                threshold = self.config["min_edge_strength"]
            
            weak_edges = []
            for source, target, edge_data in self.graph.edges(data="edge_data"):
                if edge_data.strength < threshold:
                    weak_edges.append((source, target))
            
            return weak_edges
        
        except Exception as e:
                self.logger.error(f"Error finding weak connections: {e}", exc_info=True)
                return []

    def prune_weak_connections(self,
                             threshold: float = None) -> int:
        """
        Remove weak edges from the graph.
        
        Args:
            threshold (float, optional): Strength threshold
            
        Returns:
            int: Number of edges removed
        """
        with self.lock:
            try:
                weak_edges = self.find_weak_connections(threshold)
                
                for source, target in weak_edges:
                    self.graph.remove_edge(source, target)
                    
                    # Update cache
                    if source in self.similarity_cache and \
                       target in self.similarity_cache[source]:
                        del self.similarity_cache[source][target]
                
                if weak_edges:
                    self.save_graph()
                
                return len(weak_edges)
            
            except Exception as e:
                self.logger.error(f"Error pruning weak connections: {e}", exc_info=True)
                return 0

    def strengthen_connection(self, source_id: str,
                            target_id: str,
                            amount: float = 0.1) -> None:
        """
        Strengthen connection between nodes.
        
        Args:
            source_id (str): Source node ID
            target_id (str): Target node ID
            amount (float): Amount to strengthen
        """
        try:
            self.update_edge_strength(source_id, target_id, amount)
        except Exception as e:
                self.logger.error(f"Error strengthening connection: {e}", exc_info=True)

    def get_strongest_connections(self, node_id: str,
                                limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get strongest connections for a node.
        
        Args:
            node_id (str): Node ID
            limit (int): Maximum number of connections
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, strength) pairs
        """
        try:
            if not self.graph.has_node(node_id):
                return []
            
            connections = []
            for _, target, edge_data in self.graph.edges(node_id, data="edge_data"):
                connections.append((target, edge_data.strength))
            
            for source, _, edge_data in self.graph.in_edges(node_id, data="edge_data"):
                connections.append((source, edge_data.strength))
            
            # Sort by strength
            connections.sort(key=lambda x: x[1], reverse=True)
            return connections[:limit]
        
        except Exception as e:
                self.logger.error(f"Error getting strongest connections: {e}", exc_info=True)
                return []

    def _calculate_initial_importance(self, content: str,
                                    metadata: Dict = None) -> float:
        """
        Calculate initial importance score for a node.
        
        Args:
            content (str): Node content
            metadata (Dict, optional): Additional metadata
            
        Returns:
            float: Importance score (0-1)
        """
        importance = 0.5  # Base importance
        
        if metadata:
            if metadata.get("user_marked_important"):
                importance += 0.3
            if metadata.get("frequently_accessed"):
                importance += 0.2
            if metadata.get("high_connectivity"):
                importance += 0.2
        
        return min(1.0, importance)

    def _create_initial_connections(self, node_id: str,
                                  embedding: List[float]) -> None:
        """
        Create initial connections for a new node.
        
        Args:
            node_id (str): Node ID
            embedding (List[float]): Node embedding
        """
        try:
            similar_nodes = self.find_related(
                embedding,
                threshold=self.config["similarity_threshold"]
            )
            
            for similar_id, similarity in similar_nodes:
                if similar_id != node_id:
                    self.add_edge(
                        node_id,
                        similar_id,
                        "similarity",
                        similarity
                    )
        
        except Exception as e:
                self.logger.error(f"Error creating initial connections: {e}", exc_info=True)

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
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}", exc_info=True)
            return 0.0

    def _update_node_access(self, node_id: str) -> None:
        """
        Update node access statistics.
        
        Args:
            node_id (str): Node ID
        """
        try:
            node = self.graph.nodes[node_id]["node_data"]
            node.last_accessed = datetime.now()
            node.access_count += 1
        except Exception as e:
            self.logger.error(f"Error updating node access: {e}", exc_info=True)

    def _update_edge_access(self, source_id: str, target_id: str) -> None:
        """
        Update edge access statistics.
        
        Args:
            source_id (str): Source node ID
            target_id (str): Target node ID
        """
        try:
            edge = self.graph[source_id][target_id]["edge_data"]
            edge.last_accessed = datetime.now()
            edge.access_count += 1
        except Exception as e:
            self.logger.error(f"Error updating edge access: {e}", exc_info=True)

    def get_graph_stats(self) -> Dict:
        """
        Get current graph statistics.
        
        Returns:
            Dict: Graph statistics
        """
        try:
            return {
                "node_count": self.graph.number_of_nodes(),
                "edge_count": self.graph.number_of_edges(),
                "average_degree": np.mean([
                    d for _, d in self.graph.degree()
                ]),
                "average_node_importance": np.mean([
                    node_data.importance
                    for _, node_data in self.graph.nodes(data="node_data")
                ]),
                "average_edge_strength": np.mean([
                    edge_data.strength
                    for _, _, edge_data in self.graph.edges(data="edge_data")
                ]),
                "connected_components": nx.number_strongly_connected_components(self.graph),
                "density": nx.density(self.graph)
            }
        
        except Exception as e:
            self.logger.error(f"Error getting graph stats: {e}", exc_info=True)
            return {}
