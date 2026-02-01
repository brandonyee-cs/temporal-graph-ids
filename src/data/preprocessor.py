"""Data preprocessing pipeline for T-GAT lateral movement detection.

This module implements the DataPreprocessor class that handles:
- Loading authentication events from LANL dataset CSV format
- Creating temporal windows with configurable size and overlap
- Extracting labels from red team ground truth
- Creating temporal train/val/test splits
"""

from typing import List, Tuple, Optional, Set
import pandas as pd
from src.data.models import AuthenticationEvent, TimeWindow


class DataPreprocessor:
    """Preprocessor for LANL authentication dataset.
    
    Transforms raw authentication events into temporal windows suitable
    for graph construction and model training.
    
    Attributes:
        window_size: Duration of each window in seconds (default: 3600 = 1 hour)
        overlap: Overlap between consecutive windows in seconds (default: 1800 = 30 min)
    """
    
    def __init__(self, window_size: int = 3600, overlap: int = 1800):
        """Initialize DataPreprocessor.
        
        Args:
            window_size: Window duration in seconds (default 1 hour)
            overlap: Overlap between windows in seconds (default 30 min)
            
        Raises:
            ValueError: If window_size <= 0 or overlap < 0 or overlap >= window_size
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= window_size:
            raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")
        
        self.window_size = window_size
        self.overlap = overlap
    
    def load_events(self, path: str) -> pd.DataFrame:
        """Load authentication events from LANL dataset CSV format.
        
        The LANL authentication dataset has the following format:
        timestamp,source_user@domain,dest_user@domain,source_computer,dest_computer,auth_type,logon_type,auth_orientation,success/failure
        
        Args:
            path: Path to the authentication CSV file
            
        Returns:
            DataFrame with parsed authentication events
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        # LANL auth.txt format columns
        column_names = [
            'timestamp',
            'source_user',
            'dest_user', 
            'source_computer',
            'dest_computer',
            'auth_type',
            'logon_type',
            'auth_orientation',
            'success'
        ]
        
        df = pd.read_csv(path, names=column_names, header=None)
        
        # Convert timestamp to numeric (LANL uses integer seconds)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # Convert success column to boolean
        df['success'] = df['success'].apply(
            lambda x: x.lower() == 'success' if isinstance(x, str) else bool(x)
        )
        
        # Clean user fields - extract just the username part before @
        df['source_user'] = df['source_user'].apply(
            lambda x: x.split('@')[0] if isinstance(x, str) and '@' in x else str(x)
        )
        
        # Drop rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        return df

    
    def create_windows(self, events: pd.DataFrame) -> List[TimeWindow]:
        """Partition events into overlapping temporal windows.
        
        Creates sliding windows with the configured window_size and overlap.
        Each event will appear in at least one window (Property 1: Window Partitioning Coverage).
        
        Args:
            events: DataFrame with authentication events (must have 'timestamp' column)
            
        Returns:
            List of TimeWindow objects containing the partitioned events
        """
        if events.empty:
            return []
        
        # Sort events by timestamp
        events = events.sort_values('timestamp').reset_index(drop=True)
        
        min_time = events['timestamp'].min()
        max_time = events['timestamp'].max()
        
        windows = []
        step = self.window_size - self.overlap
        
        # Create windows starting from min_time
        current_start = min_time
        
        while current_start <= max_time:
            current_end = current_start + self.window_size
            
            # Filter events in this window
            mask = (events['timestamp'] >= current_start) & (events['timestamp'] < current_end)
            window_events = events[mask]
            
            # Convert DataFrame rows to AuthenticationEvent objects
            auth_events = []
            for _, row in window_events.iterrows():
                auth_event = AuthenticationEvent(
                    timestamp=float(row['timestamp']),
                    source_user=str(row['source_user']),
                    source_computer=str(row['source_computer']),
                    dest_computer=str(row['dest_computer']),
                    auth_type=str(row.get('auth_type', 'Unknown')),
                    logon_type=str(row.get('logon_type', 'Unknown')),
                    success=bool(row.get('success', True))
                )
                auth_events.append(auth_event)
            
            window = TimeWindow(
                start_time=current_start,
                end_time=current_end,
                events=auth_events,
                label=0  # Labels assigned separately via extract_labels
            )
            windows.append(window)
            
            current_start += step
        
        return windows
    
    def extract_labels(
        self, 
        windows: List[TimeWindow], 
        redteam_path: str
    ) -> List[TimeWindow]:
        """Extract binary labels from red team ground truth file.
        
        Labels windows as 1 (lateral movement) if they contain any red team events,
        0 otherwise (Property 4: Label Extraction Correctness).
        
        The LANL red team file format:
        timestamp,source_user@domain,source_computer,dest_computer
        
        Args:
            windows: List of TimeWindow objects to label
            redteam_path: Path to red team ground truth CSV file
            
        Returns:
            List of TimeWindow objects with updated labels
        """
        # Load red team events
        redteam_columns = ['timestamp', 'source_user', 'source_computer', 'dest_computer']
        redteam_df = pd.read_csv(redteam_path, names=redteam_columns, header=None)
        redteam_df['timestamp'] = pd.to_numeric(redteam_df['timestamp'], errors='coerce')
        redteam_df = redteam_df.dropna(subset=['timestamp'])
        
        # Clean source_user field
        redteam_df['source_user'] = redteam_df['source_user'].apply(
            lambda x: x.split('@')[0] if isinstance(x, str) and '@' in x else str(x)
        )
        
        # Create a set of red team event signatures for fast lookup
        # Signature: (timestamp, source_computer, dest_computer)
        redteam_signatures: Set[Tuple[float, str, str]] = set()
        for _, row in redteam_df.iterrows():
            sig = (float(row['timestamp']), str(row['source_computer']), str(row['dest_computer']))
            redteam_signatures.add(sig)
        
        # Also create time-based lookup for efficiency
        redteam_times = set(redteam_df['timestamp'].values)
        
        # Label each window
        labeled_windows = []
        for window in windows:
            is_malicious = False
            
            for event in window.events:
                # Check if this event matches a red team event
                sig = (event.timestamp, event.source_computer, event.dest_computer)
                if sig in redteam_signatures:
                    is_malicious = True
                    break
                    
                # Also check by timestamp alone (less strict matching)
                if event.timestamp in redteam_times:
                    is_malicious = True
                    break
            
            # Create new window with updated label
            labeled_window = TimeWindow(
                start_time=window.start_time,
                end_time=window.end_time,
                events=window.events,
                label=1 if is_malicious else 0
            )
            labeled_windows.append(labeled_window)
        
        return labeled_windows

    
    def split_temporal(
        self, 
        windows: List[TimeWindow],
        train_days: Tuple[int, int] = (1, 40),
        val_days: Tuple[int, int] = (41, 49),
        test_days: Tuple[int, int] = (50, 58)
    ) -> Tuple[List[TimeWindow], List[TimeWindow], List[TimeWindow]]:
        """Create temporal train/val/test splits.
        
        Splits windows based on their timestamps to prevent data leakage.
        All training timestamps < validation timestamps < test timestamps
        (Property 5: Temporal Split Ordering).
        
        Args:
            windows: List of TimeWindow objects to split
            train_days: Tuple of (start_day, end_day) for training (inclusive)
            val_days: Tuple of (start_day, end_day) for validation (inclusive)
            test_days: Tuple of (start_day, end_day) for testing (inclusive)
            
        Returns:
            Tuple of (train_windows, val_windows, test_windows)
        """
        if not windows:
            return [], [], []
        
        # Sort windows by start time
        sorted_windows = sorted(windows, key=lambda w: w.start_time)
        
        # Find the minimum timestamp to establish day 1
        min_timestamp = min(w.start_time for w in sorted_windows)
        
        # Seconds per day
        seconds_per_day = 86400
        
        # Calculate day boundaries
        train_start = min_timestamp + (train_days[0] - 1) * seconds_per_day
        train_end = min_timestamp + train_days[1] * seconds_per_day
        
        val_start = min_timestamp + (val_days[0] - 1) * seconds_per_day
        val_end = min_timestamp + val_days[1] * seconds_per_day
        
        test_start = min_timestamp + (test_days[0] - 1) * seconds_per_day
        test_end = min_timestamp + test_days[1] * seconds_per_day
        
        train_windows = []
        val_windows = []
        test_windows = []
        
        for window in sorted_windows:
            # Assign window based on its start time
            if train_start <= window.start_time < train_end:
                train_windows.append(window)
            elif val_start <= window.start_time < val_end:
                val_windows.append(window)
            elif test_start <= window.start_time < test_end:
                test_windows.append(window)
        
        return train_windows, val_windows, test_windows
    
    def events_to_dataframe(self, events: List[AuthenticationEvent]) -> pd.DataFrame:
        """Convert list of AuthenticationEvent objects to DataFrame.
        
        Utility method for converting events back to DataFrame format.
        
        Args:
            events: List of AuthenticationEvent objects
            
        Returns:
            DataFrame with event data
        """
        if not events:
            return pd.DataFrame(columns=[
                'timestamp', 'source_user', 'source_computer', 
                'dest_computer', 'auth_type', 'logon_type', 'success'
            ])
        
        data = []
        for event in events:
            data.append({
                'timestamp': event.timestamp,
                'source_user': event.source_user,
                'source_computer': event.source_computer,
                'dest_computer': event.dest_computer,
                'auth_type': event.auth_type,
                'logon_type': event.logon_type,
                'success': event.success
            })
        
        return pd.DataFrame(data)
