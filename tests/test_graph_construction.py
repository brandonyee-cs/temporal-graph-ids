"""Property-based tests for graph construction.

This module contains property-based tests using Hypothesis to verify:
- Graph construction preserves events (Property 2)
- Feature vector completeness (Property 3)
"""

from typing import List

import torch
import pytest
from hypothesis import given, settings, strategies as st, Phase

from src.data.models import AuthenticationEvent, TimeWindow
from src.data.graph_constructor import GraphConstructor


# Strategies for generating test data
@st.composite
def auth_event_strategy(draw):
    """Generate a random AuthenticationEvent."""
    timestamp = draw(st.floats(min_value=1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False))
    source_user = f"U{draw(st.integers(min_value=0, max_value=9))}"
    source_computer = f"C{draw(st.integers(min_value=0, max_value=9))}"
    dest_computer = f"C{draw(st.integers(min_value=0, max_value=9))}"
    auth_type = draw(st.sampled_from(["Kerberos", "NTLM", "Negotiate"]))
    logon_type = draw(st.sampled_from(["Interactive", "Network", "Batch"]))
    success = draw(st.booleans())
    
    return AuthenticationEvent(
        timestamp=timestamp,
        source_user=source_user,
        source_computer=source_computer,
        dest_computer=dest_computer,
        auth_type=auth_type,
        logon_type=logon_type,
        success=success
    )


@st.composite
def time_window_strategy(draw, min_events: int = 1, max_events: int = 10):
    """Generate a TimeWindow with random events."""
    num_events = draw(st.integers(min_value=min_events, max_value=max_events))
    events = [draw(auth_event_strategy()) for _ in range(num_events)]
    
    if events:
        start_time = min(e.timestamp for e in events) - 100
        end_time = max(e.timestamp for e in events) + 100
    else:
        start_time = 1000.0
        end_time = 2000.0
    
    label = draw(st.integers(min_value=0, max_value=1))
    
    return TimeWindow(
        start_time=start_time,
        end_time=end_time,
        events=events,
        label=label
    )


class TestGraphConstruction:
    """Property tests for graph construction.
    
    Feature: temporal-gat-lateral-movement, Property 2: Graph Construction Preserves Events
    Validates: Requirements 1.2, 1.3
    """

    @given(window=time_window_strategy(min_events=1, max_events=10))
    @settings(max_examples=20, deadline=None, phases=[Phase.generate])
    def test_edge_count_equals_event_count(self, window: TimeWindow):
        """
        Property 2: Graph Construction Preserves Events
        
        For any temporal window containing N authentication events, the 
        constructed graph SHALL have exactly N edges.
        
        **Validates: Requirements 1.2, 1.3**
        """
        constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=32)
        
        graph = constructor.build_graph(window)
        
        # Edge count should equal event count
        num_edges = graph.edge_index.shape[1]
        num_events = len(window.events)
        
        assert num_edges == num_events, \
            f"Edge count {num_edges} != Event count {num_events}"

    @given(window=time_window_strategy(min_events=1, max_events=10))
    @settings(max_examples=20, deadline=None, phases=[Phase.generate])
    def test_nodes_equal_unique_entities(self, window: TimeWindow):
        """
        Property 2: Graph Construction Preserves Events
        
        For any temporal window, the set of nodes SHALL equal the union of 
        all unique source users, source computers, and destination computers.
        
        **Validates: Requirements 1.2, 1.3**
        """
        constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=32)
        
        graph = constructor.build_graph(window)
        
        # Get expected unique entities
        expected_nodes = window.get_nodes()
        
        # Graph should have same number of nodes
        num_nodes = graph.x.shape[0]
        
        assert num_nodes == len(expected_nodes), \
            f"Node count {num_nodes} != Expected unique entities {len(expected_nodes)}"

    def test_empty_window_returns_empty_graph(self):
        """Test that empty window returns graph with no nodes/edges."""
        constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=32)
        
        empty_window = TimeWindow(
            start_time=1000.0,
            end_time=2000.0,
            events=[],
            label=0
        )
        
        graph = constructor.build_graph(empty_window)
        
        assert graph.x.shape[0] == 0, "Empty window should have 0 nodes"
        assert graph.edge_index.shape[1] == 0, "Empty window should have 0 edges"
        assert graph.edge_attr.shape[0] == 0, "Empty window should have 0 edge features"

    def test_graph_label_matches_window_label(self):
        """Test that graph label matches window label."""
        constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=32)
        
        for label in [0, 1]:
            window = TimeWindow(
                start_time=1000.0,
                end_time=2000.0,
                events=[
                    AuthenticationEvent(
                        timestamp=1500.0,
                        source_user="U1",
                        source_computer="C1",
                        dest_computer="C2",
                        auth_type="Kerberos",
                        logon_type="Network",
                        success=True
                    )
                ],
                label=label
            )
            
            graph = constructor.build_graph(window)
            
            assert graph.y.item() == label, f"Graph label {graph.y.item()} != Window label {label}"



class TestFeatureCompleteness:
    """Property tests for feature vector completeness.
    
    Feature: temporal-gat-lateral-movement, Property 3: Feature Vector Completeness
    Validates: Requirements 1.4, 1.5
    """

    @given(window=time_window_strategy(min_events=1, max_events=10))
    @settings(max_examples=20, deadline=None, phases=[Phase.generate])
    def test_node_features_correct_dimensions(self, window: TimeWindow):
        """
        Property 3: Feature Vector Completeness
        
        For any node in a constructed graph, the feature vector SHALL have 
        the expected dimensionality.
        
        **Validates: Requirements 1.4, 1.5**
        """
        node_feature_dim = 64
        constructor = GraphConstructor(node_feature_dim=node_feature_dim, edge_feature_dim=32)
        
        graph = constructor.build_graph(window)
        
        # Check node feature dimensions
        assert graph.x.shape[1] == node_feature_dim, \
            f"Node feature dim {graph.x.shape[1]} != Expected {node_feature_dim}"

    @given(window=time_window_strategy(min_events=1, max_events=10))
    @settings(max_examples=20, deadline=None, phases=[Phase.generate])
    def test_edge_features_correct_dimensions(self, window: TimeWindow):
        """
        Property 3: Feature Vector Completeness
        
        For any edge in a constructed graph, the feature vector SHALL have 
        the expected dimensionality.
        
        **Validates: Requirements 1.4, 1.5**
        """
        edge_feature_dim = 32
        constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=edge_feature_dim)
        
        graph = constructor.build_graph(window)
        
        # Check edge feature dimensions
        assert graph.edge_attr.shape[1] == edge_feature_dim, \
            f"Edge feature dim {graph.edge_attr.shape[1]} != Expected {edge_feature_dim}"

    @given(window=time_window_strategy(min_events=1, max_events=10))
    @settings(max_examples=20, deadline=None, phases=[Phase.generate])
    def test_no_nan_in_node_features(self, window: TimeWindow):
        """
        Property 3: Feature Vector Completeness
        
        For any node in a constructed graph, the feature vector SHALL 
        contain no NaN values.
        
        **Validates: Requirements 1.4, 1.5**
        """
        constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=32)
        
        graph = constructor.build_graph(window)
        
        # Check for NaN values in node features
        assert not torch.isnan(graph.x).any(), "Node features contain NaN values"

    @given(window=time_window_strategy(min_events=1, max_events=10))
    @settings(max_examples=20, deadline=None, phases=[Phase.generate])
    def test_no_nan_in_edge_features(self, window: TimeWindow):
        """
        Property 3: Feature Vector Completeness
        
        For any edge in a constructed graph, the feature vector SHALL 
        contain no NaN values.
        
        **Validates: Requirements 1.4, 1.5**
        """
        constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=32)
        
        graph = constructor.build_graph(window)
        
        # Check for NaN values in edge features
        assert not torch.isnan(graph.edge_attr).any(), "Edge features contain NaN values"

    @given(window=time_window_strategy(min_events=1, max_events=10))
    @settings(max_examples=20, deadline=None, phases=[Phase.generate])
    def test_no_inf_in_features(self, window: TimeWindow):
        """
        Property 3: Feature Vector Completeness
        
        For any node or edge in a constructed graph, the feature vector 
        SHALL contain no infinite values.
        
        **Validates: Requirements 1.4, 1.5**
        """
        constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=32)
        
        graph = constructor.build_graph(window)
        
        # Check for infinite values
        assert not torch.isinf(graph.x).any(), "Node features contain infinite values"
        assert not torch.isinf(graph.edge_attr).any(), "Edge features contain infinite values"

    def test_temporal_features_derived_correctly(self):
        """Test that temporal features are correctly derived from timestamps."""
        constructor = GraphConstructor(node_feature_dim=64, edge_feature_dim=32)
        
        # Create event at known timestamp (noon on a Monday)
        # 86400 * 1 = Monday (day 1), + 12 * 3600 = noon
        timestamp = 86400 + 12 * 3600  # Monday noon
        
        event = AuthenticationEvent(
            timestamp=timestamp,
            source_user="U1",
            source_computer="C1",
            dest_computer="C2",
            auth_type="Kerberos",
            logon_type="Network",
            success=True
        )
        
        window = TimeWindow(
            start_time=timestamp - 100,
            end_time=timestamp + 100,
            events=[event],
            label=0
        )
        
        graph = constructor.build_graph(window)
        
        # Edge features should exist and have correct shape
        assert graph.edge_attr.shape[0] == 1
        assert graph.edge_attr.shape[1] == 32
        
        # Time of day should be around 0.5 (noon)
        time_of_day = graph.edge_attr[0, 0].item()
        assert 0.4 < time_of_day < 0.6, f"Time of day {time_of_day} not around 0.5 for noon"
