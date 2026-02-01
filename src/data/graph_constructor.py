"""Graph construction module for T-GAT lateral movement detection.

This module implements the GraphConstructor class that handles:
- Building PyTorch Geometric Data objects from temporal windows
- Computing node features with historical patterns and role indicators
- Computing edge features with temporal information
"""

from typing import Dict, List, Optional, Set, Tuple
import math
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.data.models import AuthenticationEvent, TimeWindow


class GraphConstructor:
    """Constructs heterogeneous graphs from authentication events.
    
    Builds PyTorch Geometric Data objects where:
    - Nodes represent computers and user accounts
    - Edges represent authentication attempts
    - Node features include historical patterns and role indicators
    - Edge features include temporal information
    
    Attributes:
        node_feature_dim: Dimension of node feature vectors
        edge_feature_dim: Dimension of edge feature vectors
    """
    
    def __init__(self, node_feature_dim: int = 64, edge_feature_dim: int = 32):
        """Initialize GraphConstructor.
        
        Args:
            node_feature_dim: Dimension of node feature vectors (default: 64)
            edge_feature_dim: Dimension of edge feature vectors (default: 32)
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        
        # Historical data storage for computing patterns
        self._node_history: Dict[str, List[float]] = {}
        self._edge_history: Dict[Tuple[str, str], List[float]] = {}
    
    def build_graph(
        self, 
        window: TimeWindow,
        history: Optional[Dict[str, List[float]]] = None
    ) -> Data:
        """Construct PyTorch Geometric Data object from window.
        
        Creates a graph where:
        - Each unique entity (user/computer) becomes a node
        - Each authentication event becomes an edge
        (Property 2: Graph Construction Preserves Events)
        
        Args:
            window: TimeWindow containing authentication events
            history: Optional historical activity data for nodes
            
        Returns:
            PyTorch Geometric Data object with:
            - x: Node features [num_nodes, node_feature_dim]
            - edge_index: Edge connectivity [2, num_edges]
            - edge_attr: Edge features [num_edges, edge_feature_dim]
            - y: Window-level label
        """
        if not window.events:
            # Return empty graph for empty windows
            return Data(
                x=torch.zeros((0, self.node_feature_dim), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, self.edge_feature_dim), dtype=torch.float32),
                y=torch.tensor([window.label], dtype=torch.long)
            )
        
        # Get unique nodes and create mapping
        nodes = list(window.get_nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Compute node features
        node_features = self.compute_node_features(nodes, window.events, history)
        
        # Build edge index and features
        edge_sources = []
        edge_targets = []
        edge_features_list = []
        
        for event in window.events:
            # Edge from source_computer to dest_computer
            src_idx = node_to_idx[event.source_computer]
            dst_idx = node_to_idx[event.dest_computer]
            
            edge_sources.append(src_idx)
            edge_targets.append(dst_idx)
            
            # Compute edge features for this event
            edge_feat = self.compute_edge_features([event], window.start_time)
            edge_features_list.append(edge_feat[0])
        
        # Stack edge features
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.stack(edge_features_list) if edge_features_list else \
                    torch.zeros((0, self.edge_feature_dim), dtype=torch.float32)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([window.label], dtype=torch.long),
            num_nodes=len(nodes)
        )

    
    def compute_node_features(
        self,
        nodes: List[str],
        events: List[AuthenticationEvent],
        history: Optional[Dict[str, List[float]]] = None
    ) -> Tensor:
        """Compute node features including historical patterns and role indicators.
        
        Features include:
        - Activity count in current window (normalized)
        - Role indicators (is_user, is_computer, is_server)
        - Historical activity patterns
        - Subnet information (encoded from node name)
        (Property 3: Feature Vector Completeness)
        
        Args:
            nodes: List of node identifiers
            events: List of authentication events in the window
            history: Optional historical activity data
            
        Returns:
            Tensor of shape [num_nodes, node_feature_dim] with no NaN values
        """
        if history is None:
            history = self._node_history
        
        # Count activity per node in current window
        activity_count: Dict[str, int] = {node: 0 for node in nodes}
        for event in events:
            activity_count[event.source_user] = activity_count.get(event.source_user, 0) + 1
            activity_count[event.source_computer] = activity_count.get(event.source_computer, 0) + 1
            activity_count[event.dest_computer] = activity_count.get(event.dest_computer, 0) + 1
        
        # Normalize activity counts
        max_activity = max(activity_count.values()) if activity_count else 1
        max_activity = max(max_activity, 1)  # Avoid division by zero
        
        features_list = []
        for node in nodes:
            features = []
            
            # Activity count (normalized)
            activity = activity_count.get(node, 0) / max_activity
            features.append(activity)
            
            # Role indicators
            is_user = 1.0 if self._is_user(node) else 0.0
            is_computer = 1.0 if self._is_computer(node) else 0.0
            is_server = 1.0 if self._is_server(node) else 0.0
            features.extend([is_user, is_computer, is_server])
            
            # Historical activity pattern (mean, std, trend)
            hist_data = history.get(node, [])
            if hist_data:
                hist_mean = sum(hist_data) / len(hist_data)
                hist_std = math.sqrt(sum((x - hist_mean) ** 2 for x in hist_data) / len(hist_data))
                # Simple trend: difference between recent and older activity
                if len(hist_data) >= 2:
                    mid = len(hist_data) // 2
                    recent = sum(hist_data[mid:]) / len(hist_data[mid:])
                    older = sum(hist_data[:mid]) / len(hist_data[:mid])
                    trend = recent - older
                else:
                    trend = 0.0
            else:
                hist_mean = 0.0
                hist_std = 0.0
                trend = 0.0
            features.extend([hist_mean, hist_std, trend])
            
            # Subnet encoding (hash-based from node name)
            subnet_encoding = self._encode_subnet(node)
            features.extend(subnet_encoding)
            
            # Pad or truncate to node_feature_dim
            current_len = len(features)
            if current_len < self.node_feature_dim:
                # Pad with zeros
                features.extend([0.0] * (self.node_feature_dim - current_len))
            elif current_len > self.node_feature_dim:
                # Truncate
                features = features[:self.node_feature_dim]
            
            features_list.append(features)
        
        # Convert to tensor and ensure no NaN values
        tensor = torch.tensor(features_list, dtype=torch.float32)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return tensor
    
    def compute_edge_features(
        self,
        events: List[AuthenticationEvent],
        window_start: float = 0.0
    ) -> List[Tensor]:
        """Compute edge features including temporal information.
        
        Features include:
        - Time of day (normalized 0-1)
        - Day of week (one-hot encoded, 7 dims)
        - Time delta from window start (normalized)
        - Authentication type encoding
        - Logon type encoding
        - Success/failure flag
        - Holiday indicator
        (Property 3: Feature Vector Completeness)
        
        Args:
            events: List of authentication events
            window_start: Start timestamp of the window for relative timing
            
        Returns:
            List of Tensors, each of shape [edge_feature_dim]
        """
        edge_features = []
        
        for event in events:
            features = []
            
            # Time of day (normalized to 0-1)
            time_of_day = (event.timestamp % 86400) / 86400.0
            features.append(time_of_day)
            
            # Day of week (one-hot, 7 dimensions)
            day_of_week = int((event.timestamp // 86400) % 7)
            day_one_hot = [0.0] * 7
            day_one_hot[day_of_week] = 1.0
            features.extend(day_one_hot)
            
            # Time delta from window start (normalized by hour)
            time_delta = (event.timestamp - window_start) / 3600.0
            features.append(min(time_delta, 1.0))  # Cap at 1.0
            
            # Authentication type encoding (hash-based, 8 dims)
            auth_hash = hash(event.auth_type) % 8
            auth_encoding = [0.0] * 8
            auth_encoding[auth_hash] = 1.0
            features.extend(auth_encoding)
            
            # Logon type encoding (hash-based, 5 dims)
            logon_hash = hash(event.logon_type) % 5
            logon_encoding = [0.0] * 5
            logon_encoding[logon_hash] = 1.0
            features.extend(logon_encoding)
            
            # Success flag
            success_flag = 1.0 if event.success else 0.0
            features.append(success_flag)
            
            # Holiday indicator (simplified - weekends as "holidays")
            is_weekend = day_of_week in [5, 6]
            features.append(1.0 if is_weekend else 0.0)
            
            # Pad or truncate to edge_feature_dim
            current_len = len(features)
            if current_len < self.edge_feature_dim:
                features.extend([0.0] * (self.edge_feature_dim - current_len))
            elif current_len > self.edge_feature_dim:
                features = features[:self.edge_feature_dim]
            
            tensor = torch.tensor(features, dtype=torch.float32)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            edge_features.append(tensor)
        
        return edge_features

    
    def _is_user(self, node: str) -> bool:
        """Check if node represents a user account.
        
        LANL dataset convention: user accounts typically start with 'U'
        
        Args:
            node: Node identifier
            
        Returns:
            True if node appears to be a user account
        """
        return node.startswith('U') or '@' in node
    
    def _is_computer(self, node: str) -> bool:
        """Check if node represents a computer.
        
        LANL dataset convention: computers typically start with 'C'
        
        Args:
            node: Node identifier
            
        Returns:
            True if node appears to be a computer
        """
        return node.startswith('C') or node.endswith('$')
    
    def _is_server(self, node: str) -> bool:
        """Check if node represents a server.
        
        Heuristic: servers often have specific naming patterns
        
        Args:
            node: Node identifier
            
        Returns:
            True if node appears to be a server
        """
        server_indicators = ['SRV', 'SERVER', 'DC', 'SQL', 'WEB', 'APP', 'DB']
        node_upper = node.upper()
        return any(ind in node_upper for ind in server_indicators)
    
    def _encode_subnet(self, node: str, encoding_dim: int = 16) -> List[float]:
        """Encode subnet information from node name.
        
        Uses hash-based encoding to create a fixed-size representation
        of the subnet/domain information embedded in node names.
        
        Args:
            node: Node identifier
            encoding_dim: Dimension of the encoding
            
        Returns:
            List of floats representing subnet encoding
        """
        # Extract potential subnet/domain info
        # LANL nodes often have format like "C1234" or "U5678@DOM1"
        if '@' in node:
            domain = node.split('@')[-1]
        else:
            # Use first few characters as pseudo-subnet
            domain = node[:min(3, len(node))]
        
        # Hash-based encoding
        encoding = [0.0] * encoding_dim
        hash_val = hash(domain)
        for i in range(min(4, encoding_dim)):
            idx = (hash_val + i * 7) % encoding_dim
            encoding[idx] = 1.0
        
        return encoding
    
    def update_history(self, window: TimeWindow) -> None:
        """Update historical activity data from a processed window.
        
        Maintains rolling history of node activity for computing
        historical pattern features.
        
        Args:
            window: Processed TimeWindow
        """
        # Count activity per node
        activity_count: Dict[str, int] = {}
        for event in window.events:
            activity_count[event.source_user] = activity_count.get(event.source_user, 0) + 1
            activity_count[event.source_computer] = activity_count.get(event.source_computer, 0) + 1
            activity_count[event.dest_computer] = activity_count.get(event.dest_computer, 0) + 1
        
        # Update history (keep last 100 windows)
        max_history = 100
        for node, count in activity_count.items():
            if node not in self._node_history:
                self._node_history[node] = []
            self._node_history[node].append(float(count))
            if len(self._node_history[node]) > max_history:
                self._node_history[node] = self._node_history[node][-max_history:]
        
        # Update edge history
        for event in window.events:
            edge_key = (event.source_computer, event.dest_computer)
            if edge_key not in self._edge_history:
                self._edge_history[edge_key] = []
            self._edge_history[edge_key].append(event.timestamp)
            if len(self._edge_history[edge_key]) > max_history:
                self._edge_history[edge_key] = self._edge_history[edge_key][-max_history:]
    
    def clear_history(self) -> None:
        """Clear all historical data."""
        self._node_history.clear()
        self._edge_history.clear()
