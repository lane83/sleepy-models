#!/usr/bin/env python3
"""
Graph operations utility for Sleepy-Models.
Provides advanced graph analysis and manipulation functions.
"""

from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import community  # python-louvain package
from dataclasses import dataclass


@dataclass
class GraphAnalysis:
    """Results of graph analysis operations."""
    centrality_scores: Dict[str, float]
    communities: Dict[str, int]
    importance_ranks: Dict[str, int]
    node_clusters: Dict[int, List[str]]
    connectivity_metrics: Dict[str, float]


class GraphOperations:
    """Advanced graph operations and analysis utilities."""
    
    def __init__(self, knowledge_graph):
        """
        Initialize graph operations.
        
        Args:
            knowledge_graph: KnowledgeGraph instance
        """
        self.knowledge_graph = knowledge_graph
        self.graph = knowledge_graph.graph
        
        # Analysis cache
        self.centrality_cache = {}
        self.community_cache = {}
        self.cache_timestamp = datetime.now()
        
        # Configuration
        self.config = {
            "cache_ttl": 3600,  # seconds
            "min_community_size": 3,
            "max_path_length": 5,
            "centrality_weight": 0.4,
            "connectivity_weight": 0.3,
            "recency_weight": 0.3
        }

    def analyze_graph_structure(self) -> GraphAnalysis:
        """
        Perform comprehensive graph analysis.
        
        Returns:
            GraphAnalysis: Analysis results
        """
        try:
            # Calculate node centrality
            centrality_scores = self._calculate_centrality()
            
            # Detect communities
            communities = self._detect_communities()
            
            # Calculate importance ranks
            importance_ranks = self._calculate_importance_ranks(
                centrality_scores,
                communities
            )
            
            # Cluster nodes
            node_clusters = self._cluster_nodes(communities)
            
            # Calculate connectivity metrics
            connectivity_metrics = self._calculate_connectivity_metrics()
            
            return GraphAnalysis(
                centrality_scores=centrality_scores,
                communities=communities,
                importance_ranks=importance_ranks,
                node_clusters=node_clusters,
                connectivity_metrics=connectivity_metrics
            )
        
        except Exception as e:
            print(f"Error analyzing graph structure: {e}")
            return GraphAnalysis(
                centrality_scores={},
                communities={},
                importance_ranks={},
                node_clusters={},
                connectivity_metrics={}
            )

    def find_key_nodes(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find most important nodes in the graph.
        
        Args:
            top_n (int): Number of nodes to return
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, importance) pairs
        """
        try:
            # Get centrality scores
            centrality = nx.pagerank(self.graph)
            
            # Combine with node importance
            scores = {}
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]["node_data"]
                scores[node_id] = (
                    self.config["centrality_weight"] * centrality[node_id] +
                    (1 - self.config["centrality_weight"]) * node_data.importance
                )
            
            # Sort by score
            key_nodes = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return key_nodes[:top_n]
        
        except Exception as e:
            print(f"Error finding key nodes: {e}")
            return []

    def find_connecting_concepts(self, node_id1: str,
                               node_id2: str,
                               max_paths: int = 3) -> List[List[str]]:
        """
        Find concepts that connect two nodes.
        
        Args:
            node_id1 (str): First node ID
            node_id2 (str): Second node ID
            max_paths (int): Maximum number of paths to return
            
        Returns:
            List[List[str]]: List of paths connecting the nodes
        """
        try:
            if not (self.graph.has_node(node_id1) and
                   self.graph.has_node(node_id2)):
                return []
            
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.graph,
                source=node_id1,
                target=node_id2,
                cutoff=self.config["max_path_length"]
            ))
            
            # Sort paths by total edge strength
            def path_strength(path):
                total_strength = 0
                for i in range(len(path) - 1):
                    edge_data = self.graph[path[i]][path[i+1]]["edge_data"]
                    total_strength += edge_data.strength
                return total_strength / (len(path) - 1)
            
            paths.sort(key=path_strength, reverse=True)
            return paths[:max_paths]
        
        except Exception as e:
            print(f"Error finding connecting concepts: {e}")
            return []

    def suggest_new_connections(self, node_id: str,
                              limit: int = 5) -> List[Tuple[str, float, str]]:
        """
        Suggest potential new connections for a node.
        
        Args:
            node_id (str): Node ID
            limit (int): Maximum number of suggestions
            
        Returns:
            List[Tuple[str, float, str]]: List of (node_id, score, reason) tuples
        """
        try:
            if not self.graph.has_node(node_id):
                return []
            
            suggestions = []
            node_data = self.graph.nodes[node_id]["node_data"]
            
            # Get current connections
            current_connections = set(self.graph.neighbors(node_id))
            
            # Find nodes with similar embeddings
            similar_nodes = self.knowledge_graph.find_related(
                node_data.embedding,
                limit=20
            )
            
            # Find nodes in same community
            communities = self._detect_communities()
            node_community = communities.get(node_id)
            community_nodes = [
                n for n, c in communities.items()
                if c == node_community and n != node_id
            ]
            
            # Score potential connections
            scores = {}
            reasons = {}
            
            for similar_id, similarity in similar_nodes:
                if similar_id not in current_connections:
                    scores[similar_id] = similarity
                    reasons[similar_id] = "Similar content"
            
            for node in community_nodes:
                if node not in current_connections:
                    if node in scores:
                        scores[node] += 0.3  # Boost community members
                    else:
                        scores[node] = 0.3
                        reasons[node] = "Same community"
            
            # Sort by score
            suggestions = [
                (node, score, reasons[node])
                for node, score in sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
            
            return suggestions[:limit]
        
        except Exception as e:
            print(f"Error suggesting connections: {e}")
            return []

    def find_concept_clusters(self, min_size: int = 3) -> List[Set[str]]:
        """
        Find clusters of closely related concepts.
        
        Args:
            min_size (int): Minimum cluster size
            
        Returns:
            List[Set[str]]: List of node clusters
        """
        try:
            # Create weighted graph for clustering
            weighted_graph = self.graph.copy()
            
            for u, v, data in weighted_graph.edges(data=True):
                edge_data = data["edge_data"]
                weighted_graph[u][v]["weight"] = edge_data.strength
            
            # Detect communities
            communities = community.best_partition(
                weighted_graph.to_undirected(),
                weight="weight"
            )
            
            # Group nodes by community
            clusters = defaultdict(set)
            for node, community_id in communities.items():
                clusters[community_id].add(node)
            
            # Filter by size
            return [
                cluster for cluster in clusters.values()
                if len(cluster) >= min_size
            ]
        
        except Exception as e:
            print(f"Error finding concept clusters: {e}")
            return []

    def analyze_node_influence(self, node_id: str) -> Dict[str, float]:
        """
        Analyze a node's influence in the graph.
        
        Args:
            node_id (str): Node ID
            
        Returns:
            Dict[str, float]: Influence metrics
        """
        try:
            if not self.graph.has_node(node_id):
                return {}
            
            # Calculate various centrality metrics
            degree_centrality = nx.degree_centrality(self.graph)[node_id]
            betweenness_centrality = nx.betweenness_centrality(self.graph)[node_id]
            pagerank = nx.pagerank(self.graph)[node_id]
            
            # Calculate local clustering coefficient
            clustering_coef = nx.clustering(self.graph, node_id)
            
            # Get node's community size
            communities = self._detect_communities()
            node_community = communities.get(node_id)
            community_size = sum(1 for n, c in communities.items()
                               if c == node_community)
            
            return {
                "degree_centrality": degree_centrality,
                "betweenness_centrality": betweenness_centrality,
                "pagerank": pagerank,
                "clustering_coefficient": clustering_coef,
                "community_size": community_size,
                "relative_community_size": community_size / self.graph.number_of_nodes()
            }
        
        except Exception as e:
            print(f"Error analyzing node influence: {e}")
            return {}

    def identify_knowledge_gaps(self) -> List[Tuple[Set[str], float]]:
        """
        Identify potential knowledge gaps in the graph.
        
        Returns:
            List[Tuple[Set[str], float]]: List of (node_group, gap_score) pairs
        """
        try:
            gaps = []
            
            # Find loosely connected components
            components = list(nx.weakly_connected_components(self.graph))
            
            for comp in components:
                if len(comp) > 1:  # Skip isolated nodes
                    subgraph = self.graph.subgraph(comp)
                    
                    # Calculate connectivity metrics
                    density = nx.density(subgraph)
                    avg_clustering = nx.average_clustering(subgraph)
                    
                    # Calculate gap score
                    gap_score = 1 - (density + avg_clustering) / 2
                    
                    if gap_score > 0.7:  # High gap score threshold
                        gaps.append((comp, gap_score))
            
            # Sort by gap score
            gaps.sort(key=lambda x: x[1], reverse=True)
            return gaps
        
        except Exception as e:
            print(f"Error identifying knowledge gaps: {e}")
            return []

    def _calculate_centrality(self) -> Dict[str, float]:
        """
        Calculate node centrality scores.
        
        Returns:
            Dict[str, float]: Node centrality scores
        """
        try:
            # Check cache
            if (self.centrality_cache and 
                self.cache_timestamp and
                (datetime.now() - self.cache_timestamp).total_seconds() <
                    self.config["cache_ttl"]):
                return self.centrality_cache
            
            # Calculate PageRank
            pagerank = nx.pagerank(self.graph)
            
            # Calculate betweenness centrality
            betweenness = nx.betweenness_centrality(self.graph)
            
            # Combine metrics
            centrality = {}
            for node in self.graph.nodes():
                centrality[node] = (
                    0.6 * pagerank[node] +
                    0.4 * betweenness[node]
                )
            
            # Update cache
            self.centrality_cache = centrality
            self.cache_timestamp = datetime.now()
            
            return centrality
        
        except Exception as e:
            print(f"Error calculating centrality: {e}")
            return {}

    def _detect_communities(self) -> Dict[str, int]:
        """
        Detect communities in the graph.
        
        Returns:
            Dict[str, int]: Node community assignments
        """
        try:
            # Check cache
            if (self.community_cache and
                (datetime.now() - self.cache_timestamp).seconds <
                    self.config["cache_ttl"]):
                return self.community_cache
            
            # Convert to undirected graph for community detection
            undirected = self.graph.to_undirected()
            
            # Detect communities
            communities = community.best_partition(undirected)
            
            # Update cache
            self.community_cache = communities
            self.cache_timestamp = datetime.now()
            
            return communities
        
        except Exception as e:
            print(f"Error detecting communities: {e}")
            return {}

    def _calculate_importance_ranks(self,
                                  centrality: Dict[str, float],
                                  communities: Dict[str, int]) -> Dict[str, int]:
        """
        Calculate node importance rankings.
        
        Args:
            centrality (Dict[str, float]): Node centrality scores
            communities (Dict[str, int]): Node community assignments
            
        Returns:
            Dict[str, int]: Node importance ranks
        """
        try:
            # Combine metrics for ranking
            scores = {}
            
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]["node_data"]
                
                # Get community size
                community_id = communities[node_id]
                community_size = sum(1 for n, c in communities.items()
                                   if c == community_id)
                
                # Calculate score components
                centrality_score = centrality.get(node_id, 0)
                connectivity_score = self.graph.degree(node_id) / self.graph.number_of_nodes()
                recency_score = 1 / (1 + (datetime.now() -
                                        node_data.last_accessed).total_seconds())
                
                # Combine scores
                scores[node_id] = (
                    self.config["centrality_weight"] * centrality_score +
                    self.config["connectivity_weight"] * connectivity_score +
                    self.config["recency_weight"] * recency_score
                )
            
            # Convert to ranks
            ranked_nodes = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            ranks = {
                node_id: rank
                for rank, (node_id, _) in enumerate(ranked_nodes, 1)
            }
            
            return ranks
        
        except Exception as e:
            print(f"Error calculating importance ranks: {e}")
            return {}

    def _cluster_nodes(self, communities: Dict[str, int]) -> Dict[int, List[str]]:
        """
        Group nodes into clusters based on communities.
        
        Args:
            communities (Dict[str, int]): Node community assignments
            
        Returns:
            Dict[int, List[str]]: Clustered nodes
        """
        try:
            clusters = defaultdict(list)
            
            for node_id, community_id in communities.items():
                clusters[community_id].append(node_id)
            
            # Sort nodes within each cluster by importance
            for community_id in clusters:
                clusters[community_id].sort(
                    key=lambda x: self.graph.nodes[x]["node_data"].importance,
                    reverse=True
                )
            
            return dict(clusters)
        
        except Exception as e:
            print(f"Error clustering nodes: {e}")
            return {}

    def _calculate_connectivity_metrics(self) -> Dict[str, float]:
        """
        Calculate graph connectivity metrics.
        
        Returns:
            Dict[str, float]: Connectivity metrics
        """
        try:
            metrics = {
                "density": nx.density(self.graph),
                "average_clustering": nx.average_clustering(self.graph),
                "average_shortest_path": nx.average_shortest_path_length(self.graph),
                "diameter": nx.diameter(self.graph),
                "reciprocity": nx.reciprocity(self.graph)
            }
            
            # Calculate component sizes
            components = list(nx.weakly_connected_components(self.graph))
            metrics["number_of_components"] = len(components)
            metrics["largest_component_size"] = max(len(c) for c in components)
            
            return metrics
        
        except Exception as e:
            print(f"Error calculating connectivity metrics: {e}")
            return {}

    def optimize_graph_structure(self) -> Tuple[int, int]:
        """
        Optimize graph structure by removing weak/redundant edges.
        
        Returns:
            Tuple[int, int]: (removed_edges, added_edges)
        """
        try:
            removed_edges = 0
            added_edges = 0
            
            # Remove weak edges
            weak_edges = self.knowledge_graph.find_weak_connections()
            for source, target in weak_edges:
                self.graph.remove_edge(source, target)
                removed_edges += 1
            
            # Find potential missing edges
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]["node_data"]
                similar_nodes = self.knowledge_graph.find_related(
                    node_data.embedding,
                    threshold=0.8  # High similarity threshold
                )
                
                for similar_id, similarity in similar_nodes:
                    if (similar_id != node and
                        not self.graph.has_edge(node, similar_id)):
                        self.knowledge_graph.add_edge(
                            node,
                            similar_id,
                            "similarity",
                            similarity
                        )
                        added_edges += 1
            
            return removed_edges, added_edges
        
        except Exception as e:
            print(f"Error optimizing graph structure: {e}")
            return 0, 0

    def merge_similar_nodes(self, similarity_threshold: float = 0.9) -> int:
        """
        Merge very similar nodes to reduce redundancy.
        
        Args:
            similarity_threshold (float): Threshold for merging
            
        Returns:
            int: Number of nodes merged
        """
        try:
            merged_count = 0
            nodes = list(self.graph.nodes())
            
            for i, node1 in enumerate(nodes):
                if node1 not in self.graph:  # Already merged
                    continue
                    
                node1_data = self.graph.nodes[node1]["node_data"]
                
                for node2 in nodes[i+1:]:
                    if node2 not in self.graph:  # Already merged
                        continue
                        
                    node2_data = self.graph.nodes[node2]["node_data"]
                    
                    # Calculate similarity
                    similarity = self.knowledge_graph._calculate_similarity(
                        node1_data.embedding,
                        node2_data.embedding
                    )
                    
                    if similarity >= similarity_threshold:
                        # Merge nodes
                        self._merge_nodes(node1, node2)
                        merged_count += 1
            
            return merged_count
        
        except Exception as e:
            print(f"Error merging similar nodes: {e}")
            return 0

    def _merge_nodes(self, primary_id: str, secondary_id: str) -> None:
        """
        Merge two nodes, keeping the primary and removing the secondary.
        
        Args:
            primary_id (str): ID of node to keep
            secondary_id (str): ID of node to merge and remove
        """
        try:
            # Get node data
            primary_data = self.graph.nodes[primary_id]["node_data"]
            secondary_data = self.graph.nodes[secondary_id]["node_data"]
            
            # Merge metadata
            primary_data.metadata.update(secondary_data.metadata)
            
            # Update importance
            primary_data.importance = max(
                primary_data.importance,
                secondary_data.importance
            )
            
            # Merge edges
            for predecessor in self.graph.predecessors(secondary_id):
                if predecessor != primary_id:
                    edge_data = self.graph[predecessor][secondary_id]["edge_data"]
                    if not self.graph.has_edge(predecessor, primary_id):
                        self.graph.add_edge(
                            predecessor,
                            primary_id,
                            edge_data=edge_data
                        )
            
            for successor in self.graph.successors(secondary_id):
                if successor != primary_id:
                    edge_data = self.graph[secondary_id][successor]["edge_data"]
                    if not self.graph.has_edge(primary_id, successor):
                        self.graph.add_edge(
                            primary_id,
                            successor,
                            edge_data=edge_data
                        )
            
            # Remove secondary node
            self.graph.remove_node(secondary_id)
            
        except Exception as e:
            print(f"Error merging nodes: {e}")

    def get_optimization_stats(self) -> Dict[str, float]:
        """
        Get statistics about graph optimization potential.
        
        Returns:
            Dict[str, float]: Optimization statistics
        """
        try:
            stats = {
                "redundant_edges": len(self.knowledge_graph.find_weak_connections()),
                "isolated_nodes": len(list(nx.isolates(self.graph))),
                "average_clustering": nx.average_clustering(self.graph),
                "density": nx.density(self.graph)
            }
            
            # Calculate potential merges
            potential_merges = 0
            nodes = list(self.graph.nodes())
            
            for i, node1 in enumerate(nodes):
                node1_data = self.graph.nodes[node1]["node_data"]
                
                similar_nodes = self.knowledge_graph.find_related(
                    node1_data.embedding,
                    threshold=0.9
                )
                potential_merges += len(similar_nodes)
            
            stats["potential_merges"] = potential_merges
            
            return stats
        
        except Exception as e:
            print(f"Error getting optimization stats: {e}")
            return {}
