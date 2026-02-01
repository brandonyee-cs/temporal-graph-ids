#!/usr/bin/env python3
"""
Download script for LANL Unified Host and Network Dataset.

The LANL dataset contains authentication events from Los Alamos National Laboratory
and is commonly used for lateral movement detection research.

Dataset URL: https://csr.lanl.gov/data/cyber1/

Note: The full dataset is large (~12GB compressed for auth.txt.gz).
This script provides options to download the full dataset or work with subsets.
"""

import argparse
import gzip
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional


# LANL dataset URLs and metadata
LANL_BASE_URL = "https://csr.lanl.gov/data/cyber1"

LANL_FILES = {
    "auth": {
        "filename": "auth.txt.gz",
        "description": "Authentication events (1.6B events, ~12GB compressed)",
        "required": True,
    },
    "redteam": {
        "filename": "redteam.txt.gz",
        "description": "Red team ground truth labels",
        "required": True,
    },
    "proc": {
        "filename": "proc.txt.gz",
        "description": "Process events (optional)",
        "required": False,
    },
    "flows": {
        "filename": "flows.txt.gz",
        "description": "Network flow events (optional)",
        "required": False,
    },
    "dns": {
        "filename": "dns.txt.gz",
        "description": "DNS lookup events (optional)",
        "required": False,
    },
}


def download_file(url: str, dest_path: Path, show_progress: bool = True) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"Downloading: {url}")
        print(f"Destination: {dest_path}")
        
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        def progress_hook(block_num, block_size, total_size):
            if show_progress and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook if show_progress else None)
        print()  # New line after progress
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\nHTTP Error {e.code}: {e.reason}")
        print("Note: LANL dataset requires accepting terms of use.")
        print("Please visit https://csr.lanl.gov/data/cyber1/ to download manually.")
        return False
    except urllib.error.URLError as e:
        print(f"\nURL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False


def decompress_gzip(gz_path: Path, output_path: Path, delete_gz: bool = False) -> bool:
    """Decompress a gzip file."""
    try:
        print(f"Decompressing: {gz_path}")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        if delete_gz:
            gz_path.unlink()
            print(f"  Deleted compressed file: {gz_path}")
        
        print(f"  Output: {output_path}")
        return True
    except Exception as e:
        print(f"Error decompressing: {e}")
        return False


def create_subset(
    input_path: Path,
    output_path: Path,
    max_lines: Optional[int] = None,
    max_days: Optional[int] = None,
    seconds_per_day: int = 86400
) -> int:
    """Create a subset of the data file."""
    print(f"Creating subset from: {input_path}")
    
    lines_written = 0
    max_timestamp = max_days * seconds_per_day if max_days else float('inf')
    
    # Handle both compressed and uncompressed input
    if input_path.suffix == '.gz':
        open_func = lambda p: gzip.open(p, 'rt', encoding='utf-8', errors='ignore')
    else:
        open_func = lambda p: open(p, 'r', encoding='utf-8', errors='ignore')
    
    with open_func(input_path) as f_in:
        with open(output_path, 'w') as f_out:
            for line in f_in:
                # Check line limit
                if max_lines and lines_written >= max_lines:
                    break
                
                # Check time limit (timestamp is first field)
                if max_days:
                    try:
                        timestamp = int(line.split(',')[0])
                        if timestamp > max_timestamp:
                            break
                    except (ValueError, IndexError):
                        continue
                
                f_out.write(line)
                lines_written += 1
                
                if lines_written % 1000000 == 0:
                    print(f"  Processed {lines_written:,} lines...")
    
    print(f"  Wrote {lines_written:,} lines to {output_path}")
    return lines_written


def verify_data_format(file_path: Path, expected_fields: int = 9) -> bool:
    """Verify the data file has the expected format."""
    print(f"Verifying format: {file_path}")
    
    if file_path.suffix == '.gz':
        open_func = lambda p: gzip.open(p, 'rt', encoding='utf-8', errors='ignore')
    else:
        open_func = lambda p: open(p, 'r', encoding='utf-8', errors='ignore')
    
    try:
        with open_func(file_path) as f:
            for i, line in enumerate(f):
                if i >= 10:  # Check first 10 lines
                    break
                fields = line.strip().split(',')
                if len(fields) < expected_fields:
                    print(f"  Warning: Line {i+1} has {len(fields)} fields, expected {expected_fields}")
                    return False
        print("  Format verified OK")
        return True
    except Exception as e:
        print(f"  Error verifying format: {e}")
        return False


def print_dataset_info():
    """Print information about the LANL dataset."""
    print("""
LANL Unified Host and Network Dataset
=====================================

This dataset contains authentication and other security events from 
Los Alamos National Laboratory's enterprise network over 58 days.

Key files for lateral movement detection:
- auth.txt.gz: Authentication events (~1.6 billion events)
  Format: time,src_user@domain,dst_user@domain,src_computer,dst_computer,auth_type,logon_type,auth_orientation,success

- redteam.txt.gz: Ground truth labels for red team (attack) activity
  Format: time,user@domain,src_computer,dst_computer

Dataset characteristics:
- 58 days of data
- ~12,000 users
- ~17,000 computers
- 749 red team events (lateral movement)

Citation:
A. D. Kent, "Unified Host and Network Data Set", in Data Science for 
Cyber-Security. World Scientific, Nov. 2015.

More info: https://csr.lanl.gov/data/cyber1/
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare LANL dataset for lateral movement detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset information
  python download_lanl.py --info
  
  # Download required files (auth + redteam)
  python download_lanl.py --output-dir data/
  
  # Download and create a small subset (first 5 days)
  python download_lanl.py --output-dir data/ --subset-days 5
  
  # Create subset from existing files
  python download_lanl.py --output-dir data/ --subset-days 5 --skip-download
  
  # Download all files including optional ones
  python download_lanl.py --output-dir data/ --all-files
"""
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data"),
        help="Output directory for downloaded files (default: data/)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset information and exit"
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download all files including optional ones (proc, flows, dns)"
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Decompress .gz files after download"
    )
    parser.add_argument(
        "--subset-days",
        type=int,
        help="Create a subset with only the first N days of data"
    )
    parser.add_argument(
        "--subset-lines",
        type=int,
        help="Create a subset with only the first N lines"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only process existing files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded file formats"
    )
    
    args = parser.parse_args()
    
    if args.info:
        print_dataset_info()
        return 0
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    # Determine which files to download
    files_to_process = {k: v for k, v in LANL_FILES.items() if v["required"] or args.all_files}
    
    # Download files
    if not args.skip_download:
        print("=" * 60)
        print("DOWNLOADING LANL DATASET")
        print("=" * 60)
        print()
        print("NOTE: The LANL dataset may require manual download.")
        print("If automatic download fails, please visit:")
        print("  https://csr.lanl.gov/data/cyber1/")
        print()
        
        for file_key, file_info in files_to_process.items():
            filename = file_info["filename"]
            url = f"{LANL_BASE_URL}/{filename}"
            dest_path = output_dir / filename
            
            if dest_path.exists():
                print(f"File already exists: {dest_path}")
                continue
            
            print(f"\n{file_info['description']}")
            success = download_file(url, dest_path)
            
            if not success and file_info["required"]:
                print(f"\nFailed to download required file: {filename}")
                print("Please download manually from https://csr.lanl.gov/data/cyber1/")
    
    # Decompress if requested
    if args.decompress:
        print("\n" + "=" * 60)
        print("DECOMPRESSING FILES")
        print("=" * 60)
        
        for file_key, file_info in files_to_process.items():
            gz_path = output_dir / file_info["filename"]
            if gz_path.exists():
                output_path = output_dir / file_info["filename"].replace('.gz', '')
                if not output_path.exists():
                    decompress_gzip(gz_path, output_path)
    
    # Create subset if requested
    if args.subset_days or args.subset_lines:
        print("\n" + "=" * 60)
        print("CREATING DATA SUBSET")
        print("=" * 60)
        
        subset_suffix = ""
        if args.subset_days:
            subset_suffix = f"_day1-{args.subset_days}"
        elif args.subset_lines:
            subset_suffix = f"_{args.subset_lines}lines"
        
        for file_key in ["auth", "redteam"]:
            # Try compressed first, then uncompressed
            input_path = output_dir / LANL_FILES[file_key]["filename"]
            if not input_path.exists():
                input_path = output_dir / LANL_FILES[file_key]["filename"].replace('.gz', '')
            
            if not input_path.exists():
                print(f"Source file not found: {input_path}")
                continue
            
            output_filename = f"{file_key}{subset_suffix}.txt"
            output_path = output_dir / output_filename
            
            create_subset(
                input_path,
                output_path,
                max_lines=args.subset_lines,
                max_days=args.subset_days
            )
    
    # Verify format if requested
    if args.verify:
        print("\n" + "=" * 60)
        print("VERIFYING FILE FORMATS")
        print("=" * 60)
        
        auth_path = output_dir / "auth.txt"
        if not auth_path.exists():
            auth_path = output_dir / "auth.txt.gz"
        if auth_path.exists():
            verify_data_format(auth_path, expected_fields=9)
        
        redteam_path = output_dir / "redteam.txt"
        if not redteam_path.exists():
            redteam_path = output_dir / "redteam.txt.gz"
        if redteam_path.exists():
            verify_data_format(redteam_path, expected_fields=4)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")
    
    print("\nNext steps:")
    print("  1. If download failed, manually download from https://csr.lanl.gov/data/cyber1/")
    print("  2. Train the model:")
    print(f"     python src/train.py --auth-file {output_dir}/auth.txt --redteam-file {output_dir}/redteam.txt")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
