#!/usr/bin/env python3
"""
Preprocess full NYC taxi data (14M trips) for training.

Input: yellow_tripdata_2009-01.parquet (raw taxi data)
Output: data/nyc_full/trips_processed.parquet (with H3 hex IDs)

Usage:
    python scripts/preprocess_full_data.py
    python scripts/preprocess_full_data.py --sample 0.5  # Use 50% of data
    python scripts/preprocess_full_data.py --resolution 9  # H3 resolution
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import h3
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess NYC taxi data')
    parser.add_argument('--input', type=str, 
                        default='/home/ubuntu/24hung.pt/spatial_queueing/yellow_tripdata_2009-01.parquet',
                        help='Input parquet file')
    parser.add_argument('--output-dir', type=str, default='data/nyc_full',
                        help='Output directory')
    parser.add_argument('--sample', type=float, default=1.0,
                        help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--resolution', type=int, default=9,
                        help='H3 resolution (7-10, default 9)')
    parser.add_argument('--chunk-size', type=int, default=500000,
                        help='Chunk size for processing')
    return parser.parse_args()


def lat_lon_to_h3(lat: float, lon: float, resolution: int) -> str:
    """Convert lat/lon to H3 hex ID."""
    try:
        return h3.latlng_to_cell(lat, lon, resolution)
    except:
        return None


def process_chunk(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Process a chunk of data."""
    # Filter valid coordinates (NYC area)
    valid = (
        (df['Start_Lon'] > -74.5) & (df['Start_Lon'] < -73.5) &
        (df['Start_Lat'] > 40.4) & (df['Start_Lat'] < 41.0) &
        (df['End_Lon'] > -74.5) & (df['End_Lon'] < -73.5) &
        (df['End_Lat'] > 40.4) & (df['End_Lat'] < 41.0) &
        (df['Trip_Distance'] > 0) &
        (df['Fare_Amt'] > 0)
    )
    df = df[valid].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Convert to H3 hex IDs
    df['pickup_hex'] = df.apply(
        lambda row: lat_lon_to_h3(row['Start_Lat'], row['Start_Lon'], resolution),
        axis=1
    )
    df['dropoff_hex'] = df.apply(
        lambda row: lat_lon_to_h3(row['End_Lat'], row['End_Lon'], resolution),
        axis=1
    )
    
    # Remove rows with invalid hex
    df = df[df['pickup_hex'].notna() & df['dropoff_hex'].notna()]
    
    # Parse datetime
    df['pickup_time'] = pd.to_datetime(df['Trip_Pickup_DateTime'])
    df['dropoff_time'] = pd.to_datetime(df['Trip_Dropoff_DateTime'])
    
    # Calculate duration
    df['duration_min'] = (df['dropoff_time'] - df['pickup_time']).dt.total_seconds() / 60
    
    # Filter unrealistic durations (< 1 min or > 180 min)
    df = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)]
    
    # Rename columns
    result = pd.DataFrame({
        'trip_id': range(len(df)),  # Will be reassigned later
        'pickup_time': df['pickup_time'],
        'dropoff_time': df['dropoff_time'],
        'pickup_hex': df['pickup_hex'],
        'dropoff_hex': df['dropoff_hex'],
        'pickup_lat': df['Start_Lat'],
        'pickup_lon': df['Start_Lon'],
        'dropoff_lat': df['End_Lat'],
        'dropoff_lon': df['End_Lon'],
        'distance_km': df['Trip_Distance'] * 1.60934,  # miles to km
        'fare': df['Fare_Amt'],
        'passengers': df['Passenger_Count'],
        'duration_min': df['duration_min'],
        'hour': df['pickup_time'].dt.hour,
        'day_of_week': df['pickup_time'].dt.dayofweek,
    })
    
    return result


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Preprocessing NYC Taxi Data ===")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Sample: {args.sample * 100:.0f}%")
    print(f"H3 Resolution: {args.resolution}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(args.input)
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")
    
    # Sample if needed
    if args.sample < 1.0:
        df = df.sample(frac=args.sample, random_state=42)
        print(f"Sampled: {len(df):,} rows ({args.sample * 100:.0f}%)")
    
    # Process in chunks
    print("\nProcessing...")
    chunks = []
    chunk_size = args.chunk_size
    
    for i in tqdm(range(0, len(df), chunk_size)):
        chunk = df.iloc[i:i + chunk_size]
        processed = process_chunk(chunk, args.resolution)
        if len(processed) > 0:
            chunks.append(processed)
    
    # Combine chunks
    result = pd.concat(chunks, ignore_index=True)
    result['trip_id'] = range(len(result))
    
    print(f"\nValid trips: {len(result):,} ({len(result) / len(df) * 100:.1f}%)")
    
    # Sort by pickup time
    result = result.sort_values('pickup_time').reset_index(drop=True)
    result['trip_id'] = range(len(result))
    
    # Save processed data
    output_path = output_dir / 'trips_processed.parquet'
    result.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Build hex cache
    print("\nBuilding hex cache...")
    all_hexes = set(result['pickup_hex']) | set(result['dropoff_hex'])
    hex_mapping = {h: i for i, h in enumerate(sorted(all_hexes))}
    
    cache_path = output_dir / 'hex_cache.json'
    with open(cache_path, 'w') as f:
        json.dump(hex_mapping, f)
    print(f"Hex cache: {len(hex_mapping):,} unique hexes -> {cache_path}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total trips: {len(result):,}")
    print(f"Unique hexes: {len(hex_mapping):,}")
    print(f"Date range: {result['pickup_time'].min()} to {result['pickup_time'].max()}")
    print(f"Avg fare: ${result['fare'].mean():.2f}")
    print(f"Avg distance: {result['distance_km'].mean():.2f} km")
    print(f"Avg duration: {result['duration_min'].mean():.1f} min")
    
    # Trips per hour
    hourly = result.groupby('hour').size()
    print(f"\nTrips by hour:")
    print(f"  Min: {hourly.min():,} (hour {hourly.idxmin()})")
    print(f"  Max: {hourly.max():,} (hour {hourly.idxmax()})")
    print(f"  Avg: {hourly.mean():,.0f}")
    
    # Calculate recommended max_trips
    steps_per_episode = 120  # 10h @ 5min
    total_steps = 31 * 24 * 12  # 31 days
    avg_trips_per_step = len(result) / total_steps
    peak_ratio = hourly.max() / hourly.mean()
    
    recommended_max = int(avg_trips_per_step * steps_per_episode * peak_ratio * 2)
    print(f"\nRecommended max_trips: {recommended_max:,}")
    
    # Save metadata
    metadata = {
        'total_trips': len(result),
        'num_hexes': len(hex_mapping),
        'h3_resolution': args.resolution,
        'date_range': [str(result['pickup_time'].min()), str(result['pickup_time'].max())],
        'avg_trips_per_step': avg_trips_per_step,
        'peak_ratio': peak_ratio,
        'recommended_max_trips': recommended_max,
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    
    print("\n=== Done! ===")


if __name__ == '__main__':
    main()
