#!/usr/bin/env python3
"""
Generate synthetic authentication data that mimics the LANL dataset format.

This allows testing the full training pipeline without downloading the real dataset.
The synthetic data includes realistic patterns and injected lateral movement attacks.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple


def generate_synthetic_lanl_data(
    output_dir: Path,
    num_days: int = 5,
    events_per_day: int = 100000,
    num_users: int = 500,
    num_computers: int = 1000,
    attack_probability: float = 0.0005,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Generate synthetic authentication data in LANL format.
    
    Returns:
        Tuple of (num_auth_events, num_redteam_events)
    """
    random.seed(seed)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate entity names
    users = [f"U{i}@DOM1" for i in range(num_users)]
    computers = [f"C{i}" for i in range(num_computers)]
    
    # Define some "normal" user-computer associations
    user_primary_computers = {user: random.choice(computers) for user in users}
    
    # Auth types and logon types from LANL
    auth_types = ["Negotiate", "Kerberos", "NTLM", "MICROSOFT_AUTHENTICATION_PACKAGE_V1_0"]
    logon_types = ["Interactive", "Network", "Batch", "Service", "NetworkCleartext", "RemoteInteractive"]
    auth_orientations = ["LogOn", "LogOff", "TGS", "TGT", "AuthMap"]
    
    auth_events = []
    redteam_events = []
    
    seconds_per_day = 86400
    
    print(f"Generating {num_days} days of synthetic data...")
    print(f"  Users: {num_users}")
    print(f"  Computers: {num_computers}")
    print(f"  Events per day: {events_per_day}")
    
    for day in range(num_days):
        day_start = day * seconds_per_day
        
        for _ in range(events_per_day):
            # Random timestamp within the day
            timestamp = day_start + random.randint(0, seconds_per_day - 1)
            
            # Select user
            src_user = random.choice(users)
            dst_user = src_user  # Usually same user
            
            # Source computer (usually user's primary)
            if random.random() < 0.8:
                src_computer = user_primary_computers[src_user]
            else:
                src_computer = random.choice(computers)
            
            # Destination computer
            if random.random() < 0.7:
                # Normal: authenticate to own computer or nearby
                dst_computer = src_computer
            else:
                dst_computer = random.choice(computers)
            
            # Auth details
            auth_type = random.choice(auth_types)
            logon_type = random.choice(logon_types)
            auth_orientation = random.choice(auth_orientations)
            success = "Success" if random.random() < 0.95 else "Fail"
            
            # Check if this should be an attack (lateral movement)
            is_attack = False
            if random.random() < attack_probability:
                # Lateral movement: unusual user accessing unusual computer
                dst_computer = random.choice(computers)
                if dst_computer != src_computer:
                    is_attack = True
                    redteam_events.append((timestamp, src_user, src_computer, dst_computer))
            
            auth_events.append((
                timestamp, src_user, dst_user, src_computer, dst_computer,
                auth_type, logon_type, auth_orientation, success
            ))
        
        print(f"  Day {day + 1}/{num_days} complete")
    
    # Sort by timestamp
    auth_events.sort(key=lambda x: x[0])
    redteam_events.sort(key=lambda x: x[0])
    
    # Write auth.txt
    auth_path = output_dir / "auth.txt"
    print(f"\nWriting {len(auth_events):,} auth events to {auth_path}")
    with open(auth_path, 'w') as f:
        for event in auth_events:
            f.write(','.join(str(x) for x in event) + '\n')
    
    # Write redteam.txt
    redteam_path = output_dir / "redteam.txt"
    print(f"Writing {len(redteam_events):,} red team events to {redteam_path}")
    with open(redteam_path, 'w') as f:
        for event in redteam_events:
            f.write(','.join(str(x) for x in event) + '\n')
    
    return len(auth_events), len(redteam_events)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic LANL-format authentication data for testing"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--num-days",
        type=int,
        default=5,
        help="Number of days to simulate (default: 5)"
    )
    parser.add_argument(
        "--events-per-day",
        type=int,
        default=100000,
        help="Authentication events per day (default: 100000)"
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=500,
        help="Number of users (default: 500)"
    )
    parser.add_argument(
        "--num-computers",
        type=int,
        default=1000,
        help="Number of computers (default: 1000)"
    )
    parser.add_argument(
        "--attack-prob",
        type=float,
        default=0.0005,
        help="Probability of attack per event (default: 0.0005)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Generate small dataset for quick testing (1 day, 10k events)"
    )
    
    args = parser.parse_args()
    
    if args.small:
        args.num_days = 1
        args.events_per_day = 10000
        args.num_users = 100
        args.num_computers = 200
    
    print("=" * 60)
    print("GENERATING SYNTHETIC LANL DATA")
    print("=" * 60)
    
    num_auth, num_redteam = generate_synthetic_lanl_data(
        output_dir=args.output_dir,
        num_days=args.num_days,
        events_per_day=args.events_per_day,
        num_users=args.num_users,
        num_computers=args.num_computers,
        attack_probability=args.attack_prob,
        seed=args.seed
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Generated {num_auth:,} authentication events")
    print(f"Generated {num_redteam:,} red team (attack) events")
    print(f"Attack ratio: {num_redteam/num_auth*100:.4f}%")
    print(f"\nFiles created in {args.output_dir}:")
    print(f"  - auth.txt")
    print(f"  - redteam.txt")
    print(f"\nTo train the model:")
    print(f"  python src/train.py --auth-file {args.output_dir}/auth.txt --redteam-file {args.output_dir}/redteam.txt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
