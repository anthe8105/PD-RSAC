"""Real trip data loader from processed parquet files."""

import torch
import pandas as pd
import h3
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime


class RealTripLoader:
    """
    Loads preprocessed NYC taxi trip data.
    
    Expects parquet with columns:
    - trip_id, pickup_time, dropoff_time
    - pickup_hex, dropoff_hex
    - pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
    - distance_km, fare, passengers, duration_min
    - hour, day_of_week
    """
    
    def __init__(
        self,
        parquet_path: str = "./data/nyc_real/trips_processed.parquet",
        device: str = "cuda",
        sample_ratio: float = 1.0,  # Sample ratio (0.0-1.0), 1.0 = use all data
        target_h3_resolution: Optional[int] = None,
        max_hex_count: Optional[int] = None,
        start_date: Optional[str] = None,  # Filter trips from this date (e.g., '2009-01-15')
        end_date: Optional[str] = None,    # Filter trips until this date (e.g., '2009-01-20')
        reference_hex_ids: Optional[List[str]] = None,  # Fixed hex set from training data
    ):
        self.parquet_path = Path(parquet_path)
        self.device = torch.device(device)
        self.sample_ratio = max(0.01, min(1.0, sample_ratio))  # Clamp to [0.01, 1.0]
        self.target_h3_resolution = target_h3_resolution
        self.max_hex_count = max_hex_count
        self.start_date = start_date
        self.end_date = end_date
        self.reference_hex_ids = reference_hex_ids
        
        self._df: Optional[pd.DataFrame] = None
        self._loaded = False
        self._hex_mapping: Optional[Dict[str, int]] = None
        self._hex_ids: Optional[List[str]] = None
        self._hex_latitudes: Optional[torch.Tensor] = None
        self._hex_longitudes: Optional[torch.Tensor] = None
        self._base_resolution: Optional[int] = None
        
        # Cached tensors for fast step-based access
        self._pickup_hexes: Optional[torch.Tensor] = None
        self._dropoff_hexes: Optional[torch.Tensor] = None
        self._fares: Optional[torch.Tensor] = None
        self._distances: Optional[torch.Tensor] = None
        
    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._loaded
    
    @property
    def total_trips(self) -> int:
        """Total number of trips."""
        if not self._loaded:
            return 0
        return len(self._df)
    
    @property
    def unique_hexes(self) -> set:
        """Set of unique hex IDs."""
        if not self._loaded:
            return set()
        return set(self._hex_ids or [])
    
    @property
    def fare_stats(self) -> Dict:
        """Fare statistics."""
        if not self._loaded:
            return {'min': 0, 'max': 0, 'mean': 0}
        return {
            'min': float(self._df['fare'].min()),
            'max': float(self._df['fare'].max()),
            'mean': float(self._df['fare'].mean()),
        }

    def _infer_hex_resolution(self) -> Optional[int]:
        """Infer resolution from the first valid hex string."""
        if self._df is None or self._df.empty:
            return None
        for column in ('pickup_hex', 'dropoff_hex'):
            series = self._df[column]
            if series.empty:
                continue
            sample_hex = series.iloc[0]
            if not isinstance(sample_hex, str):
                continue
            try:
                return h3.get_resolution(sample_hex)
            except Exception:
                continue
        return None

    def _should_coarsen_hexes(self) -> bool:
        if self.target_h3_resolution is None or self._base_resolution is None:
            return False
        return self.target_h3_resolution < self._base_resolution

    def _coarsen_hex_columns(self) -> None:
        target = int(self.target_h3_resolution)
        concat_series = pd.concat(
            [self._df['pickup_hex'], self._df['dropoff_hex']],
            ignore_index=True
        )
        unique_hexes = concat_series.unique()
        mapping = {hex_id: self._coarsen_hex(hex_id, target) for hex_id in unique_hexes}
        before_hexes = len(mapping)
        self._df['pickup_hex'] = self._df['pickup_hex'].map(mapping)
        self._df['dropoff_hex'] = self._df['dropoff_hex'].map(mapping)
        after_hexes = len(set(mapping.values()))
        print(f"Coarsened hexes from res{self._base_resolution} to res{target} ({before_hexes:,} → {after_hexes:,})")

    @staticmethod
    def _coarsen_hex(hex_id: str, target_resolution: int) -> str:
        try:
            current_res = h3.get_resolution(hex_id)
            if current_res <= target_resolution:
                return hex_id
            # h3 v4.x uses cell_to_parent instead of h3_to_parent
            return h3.cell_to_parent(hex_id, target_resolution)
        except Exception:
            return hex_id

    def _limit_hex_count(self) -> None:
        if self.max_hex_count is None or self.max_hex_count <= 0:
            return
        unique_hexes = set(self._df['pickup_hex']) | set(self._df['dropoff_hex'])
        if len(unique_hexes) <= self.max_hex_count:
            return
        pickup_counts = self._df['pickup_hex'].value_counts()
        drop_counts = self._df['dropoff_hex'].value_counts()
        combined = pickup_counts.add(drop_counts, fill_value=0)
        top_hexes = combined.nlargest(self.max_hex_count).index
        keep_mask = self._df['pickup_hex'].isin(top_hexes) & self._df['dropoff_hex'].isin(top_hexes)
        before = len(self._df)
        self._df = self._df[keep_mask].reset_index(drop=True)
        removed = before - len(self._df)
        print(f"Filtered to top {self.max_hex_count} hexes (removed {removed:,} trips)")
        
    def load(self) -> pd.DataFrame:
        """Load parquet data into memory."""
        if self._loaded and self._df is not None:
            return self._df
        
        self._df = pd.read_parquet(self.parquet_path)
        original_count = len(self._df)
        
        # Parse datetime if needed (do this first before filtering)
        if not pd.api.types.is_datetime64_any_dtype(self._df['pickup_time']):
            self._df['pickup_time'] = pd.to_datetime(self._df['pickup_time'])
        if not pd.api.types.is_datetime64_any_dtype(self._df['dropoff_time']):
            self._df['dropoff_time'] = pd.to_datetime(self._df['dropoff_time'])
        
        # Filter by date range if specified
        if self.start_date or self.end_date:
            before_filter = len(self._df)
            if self.start_date:
                start_dt = pd.to_datetime(self.start_date)
                self._df = self._df[self._df['pickup_time'] >= start_dt]
                print(f"Filtered trips from {self.start_date}: {before_filter:,} → {len(self._df):,}")
            if self.end_date:
                end_dt = pd.to_datetime(self.end_date) + pd.Timedelta(days=1)  # Include entire end_date
                before_end = len(self._df)
                self._df = self._df[self._df['pickup_time'] < end_dt]
                print(f"Filtered trips until {self.end_date}: {before_end:,} → {len(self._df):,}")
            self._df = self._df.reset_index(drop=True)
        
        # Store count after date filtering (before sampling) for accurate reporting
        trips_after_date_filter = len(self._df)
        
        # Sample data if sample_ratio < 1.0
        if self.sample_ratio < 1.0:
            self._df = self._df.sample(frac=self.sample_ratio, random_state=42)
            self._df = self._df.sort_values('pickup_time').reset_index(drop=True)
            print(f"Sampled {self.sample_ratio*100:.0f}% of {trips_after_date_filter:,} trips → {len(self._df):,} trips")
        else:
            print(f"Using all {len(self._df):,} trips (no sampling)")
        
        # Normalize hex columns to string for h3 processing
        self._df['pickup_hex'] = self._df['pickup_hex'].astype(str)
        self._df['dropoff_hex'] = self._df['dropoff_hex'].astype(str)
        
        self._base_resolution = self._infer_hex_resolution()
        if self._should_coarsen_hexes():
            self._coarsen_hex_columns()
        if self.max_hex_count is not None:
            self._limit_hex_count()
        
        # Create hex index mapping (original hex ID -> contiguous index)
        if self.reference_hex_ids is not None:
            # Use fixed hex set from training — drop trips outside it
            ref_set = set(self.reference_hex_ids)
            before = len(self._df)
            self._df = self._df[
                self._df['pickup_hex'].isin(ref_set) & self._df['dropoff_hex'].isin(ref_set)
            ].reset_index(drop=True)
            dropped = before - len(self._df)
            print(f"[Reference Hex] Using {len(self.reference_hex_ids)} reference hexes, "
                  f"dropped {dropped:,} trips outside reference set ({len(self._df):,} remaining)")
            sorted_hexes = sorted(self.reference_hex_ids)
        else:
            all_hexes = set(self._df['pickup_hex']) | set(self._df['dropoff_hex'])
            sorted_hexes = sorted(all_hexes)
        self._hex_mapping = {h: i for i, h in enumerate(sorted_hexes)}
        self._hex_ids = sorted_hexes
        self._reverse_hex_mapping = {i: h for h, i in self._hex_mapping.items()}

        # Map hex IDs to indices
        self._df['pickup_hex_idx'] = self._df['pickup_hex'].map(self._hex_mapping)
        self._df['dropoff_hex_idx'] = self._df['dropoff_hex'].map(self._hex_mapping)
        
        # Pre-cache tensors for fast access
        self._pickup_hexes = torch.tensor(
            self._df['pickup_hex_idx'].values, dtype=torch.long, device=self.device
        )
        self._dropoff_hexes = torch.tensor(
            self._df['dropoff_hex_idx'].values, dtype=torch.long, device=self.device
        )
        self._fares = torch.tensor(
            self._df['fare'].values, dtype=torch.float32, device=self.device
        )
        self._distances = torch.tensor(
            self._df['distance_km'].values, dtype=torch.float32, device=self.device
        )
        self._hex_latitudes = None
        self._hex_longitudes = None
        
        self._loaded = True
        
        # Summary message (already printed during sampling step above)
        # Just print final hex count
        print(f"Loaded {len(self._df):,} trips with {len(self._hex_mapping)} unique hexes")
        
        return self._df
    
    @property
    def num_hexes(self) -> int:
        """Number of unique hexes in the data."""
        if not self._loaded:
            self.load()
        return len(self._hex_mapping)
    
    @property
    def hex_mapping(self) -> Dict[str, int]:
        """Mapping from original hex ID to contiguous index."""
        if not self._loaded:
            self.load()
        return self._hex_mapping

    @property
    def hex_ids(self) -> List[str]:
        """Ordered list of hex IDs aligned with indices."""
        if not self._loaded:
            self.load()
        return self._hex_ids or []
    
    def get_trips_for_step(
        self,
        step: int,
        trips_per_step: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get trips for a simulation step (sequential batch).
        
        This method is optimized for training - it returns sequential batches
        of trips from the dataset, cycling through all trips.
        
        Args:
            step: Current step index (used to determine batch position)
            trips_per_step: Number of trips to return per step
            
        Returns:
            (pickup_hexes, dropoff_hexes, fares, distances) - all GPU tensors
        """
        if not self._loaded:
            self.load()
        
        total = len(self._pickup_hexes)
        start_idx = (step * trips_per_step) % total
        end_idx = min(start_idx + trips_per_step, total)
        
        # Handle wrap-around
        if end_idx - start_idx < trips_per_step and start_idx + trips_per_step > total:
            # Need to wrap around
            indices = torch.cat([
                torch.arange(start_idx, total, device=self.device),
                torch.arange(0, trips_per_step - (total - start_idx), device=self.device)
            ])
        else:
            indices = torch.arange(start_idx, end_idx, device=self.device)
        
        return (
            self._pickup_hexes[indices],
            self._dropoff_hexes[indices],
            self._fares[indices],
            self._distances[indices],
        )
    
    def get_trips_for_episode_step(
        self,
        episode_start_idx: int,
        step: int,
        step_duration_minutes: float = 5.0,
        episode_duration_hours: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get trips for a specific step within an episode, using TIME-BASED sampling.
        
        This method samples trips based on their actual pickup time, ensuring
        each episode represents a continuous 10-hour window of real data.
        
        Args:
            episode_start_idx: Starting trip index for this episode
            step: Step within episode (0 to 119 for 10h @ 5min steps)
            step_duration_minutes: Duration of each step in minutes
            episode_duration_hours: Total episode duration in hours
            
        Returns:
            (pickup_hexes, dropoff_hexes, fares, distances) - all GPU tensors
        """
        if not self._loaded:
            self.load()
        
        # Calculate time window for this step (in minutes from episode start)
        step_start_min = step * step_duration_minutes
        step_end_min = step_start_min + step_duration_minutes
        
        # Ensure time tensors are computed
        self._ensure_time_tensors()
        
        # Get the absolute start minute for this episode
        episode_start_minute = self._pickup_absolute_minutes[episode_start_idx].item()
        
        # Absolute time range for this step
        abs_start_min = episode_start_minute + step_start_min
        abs_end_min = episode_start_minute + step_end_min
        
        # Filter trips within this absolute time window
        time_mask = (
            (self._pickup_absolute_minutes >= abs_start_min) &
            (self._pickup_absolute_minutes < abs_end_min)
        )
        
        indices = time_mask.nonzero(as_tuple=True)[0]
        
        if len(indices) == 0:
            # Return empty tensors if no trips
            return (
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
            )
        
        return (
            self._pickup_hexes[indices],
            self._dropoff_hexes[indices],
            self._fares[indices],
            self._distances[indices],
        )
    
    def get_episode_start_indices(
        self,
        episode_duration_hours: float = 10.0,
        max_episodes: int = 100,
    ) -> torch.Tensor:
        """
        Get valid starting indices for episodes.
        
        Returns indices where there's enough data for a full episode.
        Useful for random episode sampling during training.
        
        Returns:
            Tensor of valid episode start indices
        """
        if not self._loaded:
            self.load()
        
        # Ensure time tensors are computed
        self._ensure_time_tensors()
        
        if len(self._pickup_absolute_minutes) == 0:
            return torch.tensor([0], dtype=torch.long, device=self.device)
            
        max_minute = self._pickup_absolute_minutes[-1].item()
        latest_start_minute = max_minute - (episode_duration_hours * 60)
        
        if latest_start_minute < 0:
            # Dataset is shorter than requested duration, just start at 0
            return torch.tensor([0], dtype=torch.long, device=self.device)
            
        valid_mask = self._pickup_absolute_minutes <= latest_start_minute
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        
        if len(valid_indices) == 0:
            return torch.tensor([0], dtype=torch.long, device=self.device)
        
        # Sample uniformly if too many
        if len(valid_indices) > max_episodes * 1000:
            step = len(valid_indices) // (max_episodes * 10)
            valid_indices = valid_indices[::step]
        
        return valid_indices
    
    def _ensure_time_tensors(self):
        """Ensure time-related tensors are computed for time-based loading."""
        if not hasattr(self, '_pickup_absolute_minutes') or self._pickup_absolute_minutes is None:
            # Compute absolute minutes from the first trip in the dataset
            # This handles multi-day, month, or even cross-year datasets gracefully
            min_time = self._df['pickup_time'].min()
            delta_minutes = (self._df['pickup_time'] - min_time).dt.total_seconds() / 60.0
            self._pickup_absolute_minutes = torch.tensor(
                delta_minutes.values,
                dtype=torch.float32, device=self.device
            )
    
    def get_trips_for_date(
        self,
        date: str,
        percentage: float = 1.0,
    ) -> pd.DataFrame:
        """Get trips for a specific date."""
        if not self._loaded:
            self.load()
        
        target_date = pd.to_datetime(date).date()
        mask = self._df['pickup_time'].dt.date == target_date
        trips = self._df[mask].copy()
        
        if percentage < 1.0:
            n = int(len(trips) * percentage)
            trips = trips.sample(n=n, random_state=42)
        
        return trips
    
    def get_trips_for_time_step(
        self,
        trips_df: pd.DataFrame,
        step: int,
        step_duration_minutes: float,
        start_hour: int = 0,
    ) -> pd.DataFrame:
        """Get trips starting in a specific time step."""
        start_minutes = start_hour * 60 + step * step_duration_minutes
        end_minutes = start_minutes + step_duration_minutes
        
        pickup_minutes = (
            trips_df['pickup_time'].dt.hour * 60 +
            trips_df['pickup_time'].dt.minute
        )
        
        mask = (pickup_minutes >= start_minutes) & (pickup_minutes < end_minutes)
        return trips_df[mask]
    
    def preprocess_episode(
        self,
        date: str,
        step_duration_minutes: float,
        episode_duration_hours: float,
        start_hour: int = 0,
        percentage: float = 1.0,
    ) -> Dict[int, pd.DataFrame]:
        """
        Preprocess trips into per-step dictionaries.
        
        Returns:
            Dict mapping step number to trips DataFrame
        """
        trips_df = self.get_trips_for_date(date, percentage)
        steps_per_episode = int(episode_duration_hours * 60 / step_duration_minutes)
        
        step_trips = {}
        for step in range(steps_per_episode):
            step_trips[step] = self.get_trips_for_time_step(
                trips_df, step, step_duration_minutes, start_hour
            )
        
        return step_trips
    
    def to_tensors(
        self,
        trips_df: pd.DataFrame,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Convert trips DataFrame to GPU tensors.
        
        Returns:
            (trip_ids, pickup_hexes, dropoff_hexes, fares, distances)
        """
        n = len(trips_df)
        
        if n == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
            )
        
        trip_ids = torch.tensor(trips_df['trip_id'].values, dtype=torch.long, device=self.device)
        pickup_hexes = torch.tensor(trips_df['pickup_hex_idx'].values, dtype=torch.long, device=self.device)
        dropoff_hexes = torch.tensor(trips_df['dropoff_hex_idx'].values, dtype=torch.long, device=self.device)
        fares = torch.tensor(trips_df['fare'].values, dtype=torch.float32, device=self.device)
        distances = torch.tensor(trips_df['distance_km'].values, dtype=torch.float32, device=self.device)
        
        return trip_ids, pickup_hexes, dropoff_hexes, fares, distances
    
    def _ensure_hex_coordinates(self) -> None:
        if self._hex_latitudes is not None and self._hex_longitudes is not None:
            return
        coords_pickup = self._df[['pickup_hex', 'pickup_lat', 'pickup_lon']].rename(
            columns={'pickup_hex': 'hex', 'pickup_lat': 'lat', 'pickup_lon': 'lon'}
        )
        coords_dropoff = self._df[['dropoff_hex', 'dropoff_lat', 'dropoff_lon']].rename(
            columns={'dropoff_hex': 'hex', 'dropoff_lat': 'lat', 'dropoff_lon': 'lon'}
        )
        coord_df = pd.concat([coords_pickup, coords_dropoff], ignore_index=True)
        if coord_df.empty:
            num_hexes = len(self._hex_mapping)
            self._hex_latitudes = torch.zeros(num_hexes, device=self.device)
            self._hex_longitudes = torch.zeros(num_hexes, device=self.device)
            return
        grouped = coord_df.groupby('hex')[['lat', 'lon']].mean()
        num_hexes = len(self._hex_mapping)
        lats = torch.zeros(num_hexes, dtype=torch.float32)
        lons = torch.zeros(num_hexes, dtype=torch.float32)

        # Try to import h3 for fallback coordinate lookup.
        # Hexes in reference_hex_ids that have zero trips in the
        # evaluation data would otherwise get (0, 0), corrupting the
        # distance matrix and GCN adjacency.
        _h3 = None
        try:
            import h3 as _h3
        except ImportError:
            pass
        n_fallback = 0
        for hex_id, idx in self._hex_mapping.items():
            if hex_id in grouped.index:
                lat, lon = grouped.loc[hex_id].values
            elif _h3 is not None:
                # Use H3 cell centre — correct for any valid H3 index
                try:
                    lat, lon = _h3.cell_to_latlng(hex_id)
                    n_fallback += 1
                except Exception:
                    lat = lon = 0.0
            else:
                lat = lon = 0.0
            lats[idx] = lat
            lons[idx] = lon
        if n_fallback > 0:
            print(f"[Coordinates] {n_fallback}/{num_hexes} hexes used H3 cell_to_latlng fallback")
        self._hex_latitudes = lats.to(self.device)
        self._hex_longitudes = lons.to(self.device)
    
    def get_hex_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get lat/lon for each hex (averaged from trip data).
        
        Returns:
            (latitudes, longitudes) tensors of shape [num_hexes]
        """
        if not self._loaded:
            self.load()
        self._ensure_hex_coordinates()
        return self._hex_latitudes, self._hex_longitudes
    
    def compute_distance_matrix(self) -> torch.Tensor:
        """
        Compute distance matrix from average trip distances.
        
        Returns:
            Distance matrix [num_hexes, num_hexes] in km
        """
        if not self._loaded:
            self.load()
        
        num_hexes = len(self._hex_mapping)
        distances = torch.zeros(num_hexes, num_hexes, device=self.device)
        
        # Use actual trip distances
        for _, row in self._df.iterrows():
            i = row['pickup_hex_idx']
            j = row['dropoff_hex_idx']
            dist = row['distance_km']
            
            # Running average
            if distances[i, j] == 0:
                distances[i, j] = dist
            else:
                distances[i, j] = 0.9 * distances[i, j] + 0.1 * dist
        
        # Make symmetric
        distances = (distances + distances.T) / 2
        
        # Fill missing with coordinate-based estimates
        lats, lons = self.get_hex_coordinates()
        for i in range(num_hexes):
            for j in range(num_hexes):
                if distances[i, j] == 0 and i != j:
                    # Haversine approximation
                    lat_diff = (lats[i] - lats[j]).abs() * 111  # km per degree
                    lon_diff = (lons[i] - lons[j]).abs() * 85   # km per degree at NYC latitude
                    distances[i, j] = torch.sqrt(lat_diff**2 + lon_diff**2)
        
        return distances
    
    def get_available_dates(self) -> List[str]:
        """Get list of available dates in the data."""
        if not self._loaded:
            self.load()
        
        dates = self._df['pickup_time'].dt.date.unique()
        return [str(d) for d in sorted(dates)]
    
    def get_stats(self) -> Dict:
        """Get statistics about the loaded data."""
        if not self._loaded:
            self.load()
        
        return {
            "total_trips": len(self._df),
            "num_hexes": len(self._hex_mapping),
            "date_range": (
                str(self._df['pickup_time'].min()),
                str(self._df['pickup_time'].max()),
            ),
            "avg_fare": self._df['fare'].mean(),
            "avg_distance": self._df['distance_km'].mean(),
            "trips_per_day": len(self._df) / self._df['pickup_time'].dt.date.nunique(),
        }
