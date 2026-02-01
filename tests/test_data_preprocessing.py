"""Property-based tests for data preprocessing pipeline.

This module contains property-based tests using Hypothesis to verify:
- Window partitioning coverage (Property 1)
- Temporal split ordering (Property 5)
- Label extraction correctness (Property 4)
"""

from typing import List

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, Phase, HealthCheck

from src.data.models import AuthenticationEvent, TimeWindow
from src.data.preprocessor import DataPreprocessor


# Simpler strategies for faster test execution
@st.composite
def simple_events_dataframe_strategy(draw, min_events: int = 1, max_events: int = 10):
    """Generate a DataFrame of authentication events with simpler generation."""
    num_events = draw(st.integers(min_value=min_events, max_value=max_events))
    
    # Generate timestamps in a reasonable range (within a week)
    base_timestamp = 1000.0
    timestamps = []
    for _ in range(num_events):
        offset = draw(st.floats(min_value=0, max_value=86400 * 7, allow_nan=False, allow_infinity=False))
        timestamps.append(base_timestamp + offset)
    
    data = []
    for i, ts in enumerate(timestamps):
        data.append({
            'timestamp': ts,
            'source_user': f"U{i % 5}",
            'source_computer': f"C{i % 3}",
            'dest_computer': f"C{(i + 1) % 3}",
            'auth_type': "Kerberos",
            'logon_type': "Network",
            'success': True
        })
    
    return pd.DataFrame(data)


@st.composite
def simple_window_config_strategy(draw):
    """Generate valid window configuration with simpler values."""
    window_size = draw(st.sampled_from([300, 600, 1800, 3600]))  # 5min, 10min, 30min, 1hr
    max_overlap = window_size - 1
    overlap = draw(st.integers(min_value=0, max_value=min(max_overlap, window_size // 2)))
    return window_size, overlap


class TestWindowPartitioning:
    """Property tests for window partitioning coverage.
    
    Feature: temporal-gat-lateral-movement, Property 1: Window Partitioning Coverage
    Validates: Requirements 1.1
    """

    @given(
        events_df=simple_events_dataframe_strategy(min_events=1, max_events=5),
        config=simple_window_config_strategy()
    )
    @settings(max_examples=20, deadline=None, phases=[Phase.generate], suppress_health_check=[HealthCheck.too_slow])
    def test_all_events_appear_in_at_least_one_window(self, events_df: pd.DataFrame, config):
        """
        Property 1: Window Partitioning Coverage
        
        For any set of authentication events and valid window configuration,
        every event SHALL fall into at least one temporal window.
        
        **Validates: Requirements 1.1**
        """
        window_size, overlap = config
        preprocessor = DataPreprocessor(window_size=window_size, overlap=overlap)
        
        windows = preprocessor.create_windows(events_df)
        
        # Collect all event timestamps from windows
        windowed_timestamps = set()
        for window in windows:
            for event in window.events:
                windowed_timestamps.add(event.timestamp)
        
        # Verify all original events appear in at least one window
        original_timestamps = set(events_df['timestamp'].values)
        
        for ts in original_timestamps:
            assert ts in windowed_timestamps, f"Event with timestamp {ts} not found in any window"

    @given(
        events_df=simple_events_dataframe_strategy(min_events=1, max_events=5),
        config=simple_window_config_strategy()
    )
    @settings(max_examples=20, deadline=None, phases=[Phase.generate], suppress_health_check=[HealthCheck.too_slow])
    def test_no_events_lost_during_partitioning(self, events_df: pd.DataFrame, config):
        """
        Property 1: Window Partitioning Coverage (no events lost)
        
        For any set of authentication events, the total count of unique events
        across all windows should equal the original event count.
        
        **Validates: Requirements 1.1**
        """
        window_size, overlap = config
        preprocessor = DataPreprocessor(window_size=window_size, overlap=overlap)
        
        windows = preprocessor.create_windows(events_df)
        
        # Count unique events across all windows (by timestamp as proxy for uniqueness)
        all_windowed_timestamps = set()
        for window in windows:
            for event in window.events:
                all_windowed_timestamps.add(event.timestamp)
        
        original_timestamps = set(events_df['timestamp'].values)
        
        # All original events should be present
        assert original_timestamps == all_windowed_timestamps, \
            f"Events lost: {original_timestamps - all_windowed_timestamps}"

    def test_empty_events_returns_empty_windows(self):
        """Test that empty DataFrame returns empty window list."""
        preprocessor = DataPreprocessor(window_size=3600, overlap=1800)
        empty_df = pd.DataFrame(columns=['timestamp', 'source_user', 'source_computer', 
                                          'dest_computer', 'auth_type', 'logon_type', 'success'])
        
        windows = preprocessor.create_windows(empty_df)
        
        assert windows == []

    def test_invalid_window_config_raises_error(self):
        """Test that invalid window configurations raise ValueError."""
        # window_size <= 0
        with pytest.raises(ValueError):
            DataPreprocessor(window_size=0, overlap=0)
        
        with pytest.raises(ValueError):
            DataPreprocessor(window_size=-100, overlap=0)
        
        # overlap < 0
        with pytest.raises(ValueError):
            DataPreprocessor(window_size=3600, overlap=-1)
        
        # overlap >= window_size
        with pytest.raises(ValueError):
            DataPreprocessor(window_size=3600, overlap=3600)
        
        with pytest.raises(ValueError):
            DataPreprocessor(window_size=3600, overlap=4000)



class TestTemporalSplitOrdering:
    """Property tests for temporal split ordering.
    
    Feature: temporal-gat-lateral-movement, Property 5: Temporal Split Ordering
    Validates: Requirements 1.7
    """

    @given(
        events_df=simple_events_dataframe_strategy(min_events=5, max_events=20)
    )
    @settings(max_examples=20, deadline=None, phases=[Phase.generate], suppress_health_check=[HealthCheck.too_slow])
    def test_train_before_val_before_test(self, events_df: pd.DataFrame):
        """
        Property 5: Temporal Split Ordering
        
        For any train/validation/test split, all training window timestamps 
        SHALL be strictly less than all validation window timestamps, and all 
        validation window timestamps SHALL be strictly less than all test timestamps.
        
        **Validates: Requirements 1.7**
        """
        preprocessor = DataPreprocessor(window_size=3600, overlap=1800)
        
        windows = preprocessor.create_windows(events_df)
        
        if len(windows) < 3:
            # Not enough windows to test split ordering
            return
        
        # Use default day splits
        train, val, test = preprocessor.split_temporal(windows)
        
        # Get max timestamp from train and min from val
        if train and val:
            max_train_time = max(w.start_time for w in train)
            min_val_time = min(w.start_time for w in val)
            assert max_train_time < min_val_time, \
                f"Train max time {max_train_time} >= Val min time {min_val_time}"
        
        # Get max timestamp from val and min from test
        if val and test:
            max_val_time = max(w.start_time for w in val)
            min_test_time = min(w.start_time for w in test)
            assert max_val_time < min_test_time, \
                f"Val max time {max_val_time} >= Test min time {min_test_time}"

    def test_empty_windows_returns_empty_splits(self):
        """Test that empty window list returns empty splits."""
        preprocessor = DataPreprocessor(window_size=3600, overlap=1800)
        
        train, val, test = preprocessor.split_temporal([])
        
        assert train == []
        assert val == []
        assert test == []

    def test_windows_sorted_by_time_in_splits(self):
        """Test that windows within each split are sorted by time."""
        # Create events spanning multiple days
        data = []
        base_time = 1000.0
        for day in range(60):  # 60 days of data
            for hour in range(24):
                ts = base_time + day * 86400 + hour * 3600
                data.append({
                    'timestamp': ts,
                    'source_user': f"U{hour % 5}",
                    'source_computer': f"C{hour % 3}",
                    'dest_computer': f"C{(hour + 1) % 3}",
                    'auth_type': "Kerberos",
                    'logon_type': "Network",
                    'success': True
                })
        
        events_df = pd.DataFrame(data)
        preprocessor = DataPreprocessor(window_size=3600, overlap=1800)
        
        windows = preprocessor.create_windows(events_df)
        train, val, test = preprocessor.split_temporal(windows)
        
        # Verify each split is sorted
        for split_name, split in [("train", train), ("val", val), ("test", test)]:
            if len(split) > 1:
                for i in range(len(split) - 1):
                    assert split[i].start_time <= split[i + 1].start_time, \
                        f"{split_name} split not sorted at index {i}"


import tempfile
import os


class TestLabelExtraction:
    """Property tests for label extraction correctness.
    
    Feature: temporal-gat-lateral-movement, Property 4: Label Extraction Correctness
    Validates: Requirements 1.6
    """

    def test_windows_with_redteam_events_labeled_1(self):
        """
        Property 4: Label Extraction Correctness
        
        For any temporal window, if the window contains at least one 
        authentication event matching a red team event, the window 
        label SHALL be 1.
        
        **Validates: Requirements 1.6**
        """
        preprocessor = DataPreprocessor(window_size=3600, overlap=1800)
        
        # Create events with known timestamps
        events = [
            AuthenticationEvent(
                timestamp=1000.0,
                source_user="U1",
                source_computer="C1",
                dest_computer="C2",
                auth_type="Kerberos",
                logon_type="Network",
                success=True
            ),
            AuthenticationEvent(
                timestamp=2000.0,
                source_user="U2",
                source_computer="C2",
                dest_computer="C3",
                auth_type="NTLM",
                logon_type="Interactive",
                success=True
            ),
            AuthenticationEvent(
                timestamp=3000.0,
                source_user="U3",
                source_computer="C3",
                dest_computer="C4",
                auth_type="Kerberos",
                logon_type="Network",
                success=True
            )
        ]
        
        windows = [
            TimeWindow(start_time=500.0, end_time=1500.0, events=[events[0]], label=0),
            TimeWindow(start_time=1500.0, end_time=2500.0, events=[events[1]], label=0),
            TimeWindow(start_time=2500.0, end_time=3500.0, events=[events[2]], label=0)
        ]
        
        # Create red team file with event at timestamp 2000.0
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("2000.0,U2@DOM1,C2,C3\n")
            redteam_path = f.name
        
        try:
            labeled_windows = preprocessor.extract_labels(windows, redteam_path)
            
            # Window containing red team event should be labeled 1
            assert labeled_windows[0].label == 0, "Window without red team event should be 0"
            assert labeled_windows[1].label == 1, "Window with red team event should be 1"
            assert labeled_windows[2].label == 0, "Window without red team event should be 0"
        finally:
            os.unlink(redteam_path)

    def test_windows_without_redteam_events_labeled_0(self):
        """
        Property 4: Label Extraction Correctness
        
        For any temporal window, if the window contains no authentication 
        events matching red team events, the window label SHALL be 0.
        
        **Validates: Requirements 1.6**
        """
        preprocessor = DataPreprocessor(window_size=3600, overlap=1800)
        
        events = [
            AuthenticationEvent(
                timestamp=1000.0,
                source_user="U1",
                source_computer="C1",
                dest_computer="C2",
                auth_type="Kerberos",
                logon_type="Network",
                success=True
            )
        ]
        
        windows = [
            TimeWindow(start_time=500.0, end_time=1500.0, events=events, label=0)
        ]
        
        # Create red team file with event at different timestamp
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("9999.0,U99@DOM1,C99,C100\n")
            redteam_path = f.name
        
        try:
            labeled_windows = preprocessor.extract_labels(windows, redteam_path)
            
            # Window should remain labeled 0
            assert labeled_windows[0].label == 0, "Window without red team event should be 0"
        finally:
            os.unlink(redteam_path)

    @given(
        num_normal_events=st.integers(min_value=1, max_value=5),
        num_redteam_events=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=20, deadline=None, phases=[Phase.generate], suppress_health_check=[HealthCheck.too_slow])
    def test_label_extraction_property(self, num_normal_events: int, num_redteam_events: int):
        """
        Property 4: Label Extraction Correctness (property-based)
        
        For any set of windows and red team events, windows containing 
        red team events SHALL be labeled 1, others SHALL be labeled 0.
        
        **Validates: Requirements 1.6**
        """
        preprocessor = DataPreprocessor(window_size=3600, overlap=1800)
        
        # Create normal events
        normal_events = []
        for i in range(num_normal_events):
            normal_events.append(AuthenticationEvent(
                timestamp=1000.0 + i * 100,
                source_user=f"U{i}",
                source_computer=f"C{i}",
                dest_computer=f"C{i+1}",
                auth_type="Kerberos",
                logon_type="Network",
                success=True
            ))
        
        # Create red team events (at different timestamps)
        redteam_timestamps = [5000.0 + i * 100 for i in range(num_redteam_events)]
        redteam_events = []
        for i, ts in enumerate(redteam_timestamps):
            redteam_events.append(AuthenticationEvent(
                timestamp=ts,
                source_user=f"REDTEAM{i}",
                source_computer=f"REDC{i}",
                dest_computer=f"REDC{i+1}",
                auth_type="Kerberos",
                logon_type="Network",
                success=True
            ))
        
        # Create windows - some with normal events, some with red team events
        windows = [
            TimeWindow(start_time=500.0, end_time=2000.0, events=normal_events, label=0),
            TimeWindow(start_time=4500.0, end_time=6000.0, events=redteam_events, label=0)
        ]
        
        # Create red team file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            for ts in redteam_timestamps:
                f.write(f"{ts},REDTEAM@DOM1,REDC0,REDC1\n")
            redteam_path = f.name
        
        try:
            labeled_windows = preprocessor.extract_labels(windows, redteam_path)
            
            # First window (normal events) should be 0
            assert labeled_windows[0].label == 0, "Window with only normal events should be 0"
            
            # Second window (red team events) should be 1
            assert labeled_windows[1].label == 1, "Window with red team events should be 1"
        finally:
            os.unlink(redteam_path)
