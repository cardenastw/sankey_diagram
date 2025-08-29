import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set, Callable
import json
import requests
from tqdm import tqdm


class FlowPathAnalyzer:
    def __init__(self, is_pre_aggregated: bool = False, show_progress: bool = True):
        self.sessions = []
        self.flow_paths = defaultdict(int)
        self.screen_transitions = defaultdict(lambda: defaultdict(int))
        self.sessions_by_source = defaultdict(list)
        self.traffic_sources = set()
        self.sessions_by_segment = defaultdict(lambda: defaultdict(list))
        self.segment_values = defaultdict(set)
        self.is_pre_aggregated = is_pre_aggregated
        self.show_progress = show_progress
        self.aggregated_paths = []  # Store pre-aggregated path data
        self.path_chunks = []  # Store chunks for chaining when pre-aggregated
        self.event_metadata = defaultdict(lambda: defaultdict(dict))  # session_id -> event_index -> metadata
        
    def load_data(self, data_source, progress_callback: Optional[Callable[[str, float], None]] = None):
        """Load data with optional progress tracking.
        
        Args:
            data_source: File path, URL, or list of data
            progress_callback: Optional callback function(message, percentage)
        """
        if progress_callback:
            progress_callback("Starting data load...", 0.0)
        elif self.show_progress:
            print("Loading data...")
        
        if isinstance(data_source, str):
            if data_source.startswith('http://') or data_source.startswith('https://'):
                if progress_callback:
                    progress_callback("Downloading from server...", 10.0)
                elif self.show_progress:
                    print("Downloading from server...")
                response = requests.get(data_source)
                response.raise_for_status()
                data = response.json()
            elif data_source.endswith('.json'):
                if progress_callback:
                    progress_callback("Reading JSON file...", 10.0)
                elif self.show_progress:
                    print("Reading JSON file...")
                with open(data_source, 'r') as f:
                    data = json.load(f)
            elif data_source.endswith('.csv'):
                if progress_callback:
                    progress_callback("Reading CSV file...", 10.0)
                elif self.show_progress:
                    print("Reading CSV file...")
                df = pd.read_csv(data_source)
                data = df.to_dict('records')
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV, or provide HTTP(S) URL.")
        elif isinstance(data_source, list):
            data = data_source
        else:
            raise ValueError("Data source must be a file path, URL, or list of dictionaries")
        
        if progress_callback:
            progress_callback("Data loaded, processing...", 30.0)
        elif self.show_progress:
            print(f"Data loaded ({len(data)} records), processing...")
        
        if self.is_pre_aggregated:
            self._process_aggregated_data(data, progress_callback)
        else:
            self._process_sessions(data, progress_callback)
        
        if progress_callback:
            progress_callback("Processing complete!", 100.0)
        elif self.show_progress:
            print("Processing complete!")
        
    def _process_sessions(self, data, progress_callback: Optional[Callable[[str, float], None]] = None):
        sessions_dict = defaultdict(list)
        session_sources = {}
        session_metadata = defaultdict(dict)
        
        # Create progress bar for event processing
        events_iterator = tqdm(data, desc="Processing events", disable=not self.show_progress) if self.show_progress else data
        total_events = len(data) if hasattr(data, '__len__') else 0
        
        for event_idx, event in enumerate(events_iterator):
            session_id = event.get('session_id', event.get('user_id', 'default'))
            screen = event.get('screen', event.get('screen_view', event.get('page')))
            timestamp = event.get('timestamp', None)
            
            # Extract traffic source information
            traffic_source = event.get('traffic_source', 
                                      event.get('source', 
                                      event.get('utm_source', 
                                      event.get('referrer', 'direct'))))
            
            # Store the first traffic source seen for this session
            if session_id not in session_sources and traffic_source:
                session_sources[session_id] = traffic_source
                self.traffic_sources.add(traffic_source)
            
            # Store all metadata fields for this session
            if session_id not in session_metadata or not session_metadata[session_id]:
                # Capture all potential segmentation fields from the first event
                for key, value in event.items():
                    if key not in ['screen', 'screen_view', 'page', 'timestamp', 'session_id', 'user_id']:
                        session_metadata[session_id][key] = value
                        self.segment_values[key].add(value)
            
            if screen:
                event_data = {
                    'screen': screen,
                    'timestamp': timestamp
                }
                # Store event-level metadata
                event_metadata = {}
                for key, value in event.items():
                    if key not in ['screen', 'screen_view', 'page', 'timestamp', 'session_id', 'user_id']:
                        event_metadata[key] = value
                        self.segment_values[key].add(value)
                
                sessions_dict[session_id].append(event_data)
                # Store event metadata indexed by session and position
                event_index = len(sessions_dict[session_id]) - 1
                self.event_metadata[session_id][event_index] = event_metadata
            
            # Update progress callback if provided
            if progress_callback and total_events > 0:
                progress = 30.0 + (event_idx / total_events) * 40.0  # 30% to 70%
                progress_callback(f"Processing events... ({event_idx + 1}/{total_events})", progress)
        
        # Process sessions into final format
        session_progress_msg = "Building session flows..."
        if progress_callback:
            progress_callback(session_progress_msg, 70.0)
        elif self.show_progress:
            print(session_progress_msg)
        
        sessions_iterator = tqdm(sessions_dict.items(), desc="Building sessions", disable=not self.show_progress) if self.show_progress else sessions_dict.items()
        total_sessions = len(sessions_dict)
        
        for session_idx, (session_id, screens) in enumerate(sessions_iterator):
            if len(screens) > 1:
                if screens[0]['timestamp']:
                    screens.sort(key=lambda x: x['timestamp'])
                
                screen_sequence = [s['screen'] for s in screens]
                traffic_source = session_sources.get(session_id, 'direct')
                
                # Build session data with all metadata
                session_data = {
                    'session_id': session_id,
                    'screens': screen_sequence,
                    'traffic_source': traffic_source
                }
                
                # Add all metadata fields
                session_data.update(session_metadata.get(session_id, {}))
                
                self.sessions.append(session_data)
                self.sessions_by_source[traffic_source].append(session_data)
                
                # Store in segment buckets for all metadata fields
                for field, value in session_metadata.get(session_id, {}).items():
                    self.sessions_by_segment[field][value].append(session_data)
                
                for i in range(len(screen_sequence) - 1):
                    from_screen = screen_sequence[i]
                    to_screen = screen_sequence[i + 1]
                    self.screen_transitions[from_screen][to_screen] += 1
            
            # Update progress callback if provided
            if progress_callback and total_sessions > 0:
                progress = 70.0 + (session_idx / total_sessions) * 25.0  # 70% to 95%
                progress_callback(f"Building sessions... ({session_idx + 1}/{total_sessions})", progress)
    
    def _process_aggregated_data(self, data, progress_callback: Optional[Callable[[str, float], None]] = None):
        """Process pre-aggregated data format as chainable chunks.
        
        Expected format:
        [
            {
                "path": ["Home", "Products", "Cart"],
                "count": 150,
                "traffic_source": "google_organic",
                "user_type": "new",
                "device": "mobile"
            }
        ]
        """
        # Create progress bar for aggregated data processing
        items_iterator = tqdm(data, desc="Processing aggregated chunks", disable=not self.show_progress) if self.show_progress else data
        total_items = len(data) if hasattr(data, '__len__') else 0
        
        # Store chunks for chaining
        self.path_chunks = []
        
        for item_idx, item in enumerate(items_iterator):
            path = item.get('path', [])
            count = item.get('count', 1)
            
            if len(path) < 2:
                continue
            
            # Store the chunk with metadata
            chunk_data = {
                'path': path,
                'count': count,
                'metadata': {k: v for k, v in item.items() if k not in ['path', 'count']}
            }
            self.path_chunks.append(chunk_data)
            
            # Store the aggregated path for compatibility
            self.aggregated_paths.append(item)
            
            # Build individual chunk paths
            path_str = ' -> '.join(path)
            self.flow_paths[path_str] += count
            
            # Build screen transitions for each step in the chunk
            for i in range(len(path) - 1):
                from_screen = path[i]
                to_screen = path[i + 1]
                self.screen_transitions[from_screen][to_screen] += count
            
            # Extract metadata for segmentation
            traffic_source = item.get('traffic_source', 'direct')
            self.traffic_sources.add(traffic_source)
            
            # Create synthetic session data for compatibility
            session_data = {
                'session_id': f"chunk_{len(self.sessions)}",
                'screens': path,
                'traffic_source': traffic_source,
                'count': count
            }
            
            # Add all metadata fields
            for key, value in item.items():
                if key not in ['path', 'count']:
                    session_data[key] = value
                    self.segment_values[key].add(value)
            
            # Store single session with count for memory efficiency
            self.sessions.append(session_data)
            self.sessions_by_source[traffic_source].append(session_data)
            
            # Store in segment buckets
            for field, value in session_data.items():
                if field not in ['session_id', 'screens', 'count']:
                    self.sessions_by_segment[field][value].append(session_data)
            
            # Update progress callback if provided
            if progress_callback and total_items > 0:
                progress = 30.0 + (item_idx / total_items) * 65.0  # 30% to 95%
                progress_callback(f"Processing aggregated chunks... ({item_idx + 1}/{total_items})", progress)
    
    def find_paths(self, start_screen: str, end_screen: str, max_length: int = 10, max_paths: int = 50) -> List[Tuple[List[str], int]]:
        if self.is_pre_aggregated:
            # Chain chunks together to find longer paths
            return self._find_chained_paths(start_screen, end_screen, max_length, max_paths)
        else:
            # Original implementation for raw event data
            paths = []
            
            for session in self.sessions:
                screens = session['screens']
                
                for i, screen in enumerate(screens):
                    if screen == start_screen:
                        for j in range(i + 1, min(i + max_length + 1, len(screens))):
                            if screens[j] == end_screen:
                                path = screens[i:j+1]
                                path_str = ' -> '.join(path)
                                self.flow_paths[path_str] += 1
                                break
            
            sorted_paths = sorted(self.flow_paths.items(), key=lambda x: x[1], reverse=True)
            
            return [(path.split(' -> '), count) for path, count in sorted_paths]
    
    def find_paths_with_filter(self, start_screen: str, end_screen: str, field_filter: Dict[str, str] = None,
                              max_length: int = 10, max_paths: int = 50) -> List[Tuple[List[str], int]]:
        """Find paths from start to end screen with field filtering.
        
        Args:
            start_screen: Starting screen name
            end_screen: Ending screen name
            field_filter: Dictionary of field->value filters to apply to sessions
            max_length: Maximum path length to consider
            max_paths: Maximum number of paths to return
            
        Returns:
            List of (path, count) tuples sorted by frequency
        """
        if self.is_pre_aggregated:
            # For pre-aggregated data, filter chunks by metadata and then chain
            if field_filter:
                # Filter chunks by metadata
                filtered_chunks = []
                for chunk in self.path_chunks:
                    matches = True
                    for field, expected_value in field_filter.items():
                        if field in chunk['metadata']:
                            chunk_value = chunk['metadata'][field]
                            if str(chunk_value).lower() != str(expected_value).lower():
                                matches = False
                                break
                    if matches:
                        filtered_chunks.append(chunk)
                
                # Temporarily replace path_chunks for chaining
                original_chunks = self.path_chunks
                self.path_chunks = filtered_chunks
                result = self._find_chained_paths(start_screen, end_screen, max_length, max_paths)
                self.path_chunks = original_chunks
                return result
            else:
                return self._find_chained_paths(start_screen, end_screen, max_length, max_paths)
        else:
            # For raw session data, filter sessions by metadata
            if field_filter:
                filtered_sessions = []
                for session in self.sessions:
                    matches = True
                    for field, expected_value in field_filter.items():
                        if field in session:
                            session_value = session[field]
                            if str(session_value).lower() != str(expected_value).lower():
                                matches = False
                                break
                    if matches:
                        filtered_sessions.append(session)
            else:
                filtered_sessions = self.sessions
            
            # Find paths in filtered sessions
            flow_paths = defaultdict(int)
            
            for session in filtered_sessions:
                screens = session['screens']
                session_count = session.get('count', 1)
                
                for i, screen in enumerate(screens):
                    if screen == start_screen:
                        for j in range(i + 1, min(i + max_length + 1, len(screens))):
                            if screens[j] == end_screen:
                                path = screens[i:j+1]
                                path_str = ' -> '.join(path)
                                flow_paths[path_str] += session_count
                                break
            
            sorted_paths = sorted(flow_paths.items(), key=lambda x: x[1], reverse=True)[:max_paths]
            return [(path.split(' -> '), count) for path, count in sorted_paths]
    
    def _find_chained_paths(self, start_screen: str, end_screen: str, max_length: int = 10, max_paths: int = 50) -> List[Tuple[List[str], int]]:
        """Chain together pre-aggregated chunks to find longer paths."""
        from collections import defaultdict, deque
        
        # Build a graph of chunk connections (optimized)
        chunks_starting_with = defaultdict(list)
        chunks_ending_with = defaultdict(list)
        
        # Organize chunks by their start and end screens
        for chunk in self.path_chunks:
            path = chunk['path']
            if len(path) < 2:
                continue
            start = path[0]
            end = path[-1]
            
            chunks_starting_with[start].append(chunk)
            chunks_ending_with[end].append(chunk)
        
        # Use BFS to find all possible chained paths
        found_paths = {}
        queue = deque()
        max_depth = min(max_length // 2, 5)  # Limit exploration depth
        
        # Start with chunks that begin with start_screen (sorted by count for better early results)
        starting_chunks = sorted(chunks_starting_with[start_screen], key=lambda x: x['count'], reverse=True)
        for chunk in starting_chunks:
            path = chunk['path']
            count = chunk['count']
            queue.append((path, count, {tuple(path)}, 1))  # path, count, visited_chunks, depth
        
        while queue:
            current_path, current_count, visited_chunks, depth = queue.popleft()
            
            if len(current_path) > max_length or depth > max_depth:
                continue
                
            # If we've reached the end screen, record this path
            if current_path[-1] == end_screen:
                path_str = ' -> '.join(current_path)
                if path_str not in found_paths:
                    found_paths[path_str] = 0
                found_paths[path_str] += current_count
                
                # Early termination: stop if we have enough paths
                if len(found_paths) >= max_paths:
                    break
                continue
            
            # Try to extend this path with compatible chunks
            current_end = current_path[-1]
            # Sort next chunks by count for better early results
            next_chunks = sorted(chunks_starting_with[current_end], key=lambda x: x['count'], reverse=True)
            for next_chunk in next_chunks:
                next_path = next_chunk['path']
                next_chunk_tuple = tuple(next_path)
                
                if next_chunk_tuple in visited_chunks:
                    continue  # Avoid infinite loops
                
                # Chain the chunks (overlap by 1 screen)
                if len(next_path) > 1:
                    chained_path = current_path + next_path[1:]  # Skip first element to avoid duplication
                    chained_count = min(current_count, next_chunk['count'])  # Use minimum count
                    new_visited = visited_chunks | {next_chunk_tuple}
                    
                    queue.append((chained_path, chained_count, new_visited, depth + 1))
        
        # Also include direct chunks that match start->end
        for path_str, count in self.flow_paths.items():
            path = path_str.split(' -> ')
            if (len(path) <= max_length and 
                path[0] == start_screen and 
                path[-1] == end_screen):
                if path_str not in found_paths:
                    found_paths[path_str] = 0
                found_paths[path_str] += count
        
        # Sort and return results
        sorted_paths = sorted(found_paths.items(), key=lambda x: x[1], reverse=True)
        return [(path.split(' -> '), count) for path, count in sorted_paths]
    
    def find_all_paths_between(self, start_screen: str, end_screen: str, 
                               max_depth: int = 5) -> List[Tuple[List[str], int]]:
        def dfs(current: str, target: str, path: List[str], visited: Set[str], depth: int):
            if depth > max_depth:
                return []
            
            if current == target:
                return [path]
            
            all_paths = []
            visited.add(current)
            
            if current in self.screen_transitions:
                for next_screen, count in self.screen_transitions[current].items():
                    if next_screen not in visited:
                        new_paths = dfs(next_screen, target, path + [next_screen], 
                                      visited.copy(), depth + 1)
                        all_paths.extend(new_paths)
            
            return all_paths
        
        unique_paths = dfs(start_screen, end_screen, [start_screen], set(), 0)
        
        path_counts = {}
        for session in self.sessions:
            screens = session['screens']
            for path in unique_paths:
                path_str = ' -> '.join(path)
                for i in range(len(screens) - len(path) + 1):
                    if screens[i:i+len(path)] == path:
                        path_counts[path_str] = path_counts.get(path_str, 0) + 1
        
        return [(path.split(' -> '), count) for path, count in 
                sorted(path_counts.items(), key=lambda x: x[1], reverse=True)]
    
    def get_most_common_paths(self, top_n: int = 10) -> List[Tuple[List[str], int]]:
        if self.is_pre_aggregated:
            # For pre-aggregated data, return the chunk paths and try to find longer paths
            # First, get all chunk paths
            chunk_paths = {}
            for path_str, count in self.flow_paths.items():
                chunk_paths[path_str] = count
            
            # Also try to find some longer chained paths by exploring common transitions
            from collections import defaultdict, deque
            
            chunks_starting_with = defaultdict(list)
            chunks_ending_with = defaultdict(list)
            
            # Organize chunks by their start and end screens
            for chunk in self.path_chunks:
                path = chunk['path']
                if len(path) >= 2:
                    start = path[0]
                    end = path[-1]
                    chunks_starting_with[start].append(chunk)
                    chunks_ending_with[end].append(chunk)
            
            # Find some longer paths by simple chaining (limit to prevent explosion)
            explored_chains = set()
            for chunk in self.path_chunks[:50]:  # Limit exploration
                if len(chunk['path']) >= 3:
                    path = chunk['path']
                    end_screen = path[-1]
                    
                    # Try to extend with one more chunk
                    for next_chunk in chunks_starting_with[end_screen][:5]:  # Limit branches
                        next_path = next_chunk['path']
                        if len(next_path) > 1:
                            chained = path + next_path[1:]  # Skip overlap
                            if len(chained) <= 6:  # Reasonable limit
                                chained_str = ' -> '.join(chained)
                                if chained_str not in explored_chains:
                                    explored_chains.add(chained_str)
                                    min_count = min(chunk['count'], next_chunk['count'])
                                    if chained_str not in chunk_paths or chunk_paths[chained_str] < min_count:
                                        chunk_paths[chained_str] = min_count
            
            # Sort and return results
            sorted_paths = sorted(chunk_paths.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return [(path.split(' -> '), count) for path, count in sorted_paths]
        else:
            # Original implementation for raw session data
            all_paths = defaultdict(int)
            
            for session in self.sessions:
                screens = session['screens']
                session_count = session.get('count', 1)
                
                for length in range(2, min(6, len(screens) + 1)):
                    for i in range(len(screens) - length + 1):
                        path = tuple(screens[i:i+length])
                        all_paths[path] += session_count
            
            sorted_paths = sorted(all_paths.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return [(list(path), count) for path, count in sorted_paths]
    
    def get_preceding_steps(self, session_screens: List[str], target_screen: str, 
                           max_steps: int = 3) -> List[str]:
        """Get the steps that preceded reaching the target screen in a session.
        
        Args:
            session_screens: The full list of screens in a session
            target_screen: The screen to find predecessors for
            max_steps: Maximum number of preceding steps to return
            
        Returns:
            List of screens that preceded the target screen
        """
        if target_screen not in session_screens:
            return []
        
        target_index = session_screens.index(target_screen)
        start_index = max(0, target_index - max_steps)
        
        return session_screens[start_index:target_index]
    
    def find_paths_with_context(self, start_screen: str, end_screen: str, 
                               max_length: int = 10, context_steps: int = 3) -> Dict:
        """Find paths from start to end, including preceding context.
        
        Args:
            start_screen: Starting screen name
            end_screen: Ending screen name
            max_length: Maximum path length between start and end
            context_steps: Number of steps before start to include as context
            
        Returns:
            Dictionary with:
                - paths: List of (path, count) tuples for start->end paths
                - context_paths: Dictionary mapping each path to its preceding context
                - full_journeys: List of complete journeys (context + path)
        """
        paths = []
        context_paths = {}
        full_journeys = []
        path_counts = defaultdict(int)
        journey_counts = defaultdict(int)
        
        for session in self.sessions:
            screens = session['screens']
            session_count = session.get('count', 1)  # Use aggregated count
            
            for i, screen in enumerate(screens):
                if screen == start_screen:
                    # Find if this start leads to the end
                    for j in range(i + 1, min(i + max_length + 1, len(screens))):
                        if screens[j] == end_screen:
                            # Found a path from start to end
                            path = screens[i:j+1]
                            path_str = ' -> '.join(path)
                            path_counts[path_str] += session_count
                            
                            # Get preceding context
                            preceding = self.get_preceding_steps(screens, start_screen, context_steps)
                            
                            if preceding:
                                context_str = ' -> '.join(preceding)
                                if path_str not in context_paths:
                                    context_paths[path_str] = defaultdict(int)
                                context_paths[path_str][context_str] += session_count
                                
                                # Create full journey
                                full_journey = preceding + path
                                journey_str = ' -> '.join(full_journey)
                                journey_counts[journey_str] += session_count
                            else:
                                # No preceding steps, just the path itself
                                journey_str = path_str
                                journey_counts[journey_str] += session_count
                            
                            break
        
        # Sort paths by frequency
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
        paths = [(path.split(' -> '), count) for path, count in sorted_paths]
        
        # Sort full journeys by frequency
        sorted_journeys = sorted(journey_counts.items(), key=lambda x: x[1], reverse=True)
        full_journeys = [(journey.split(' -> '), count) for journey, count in sorted_journeys]
        
        return {
            'paths': paths,
            'context_paths': context_paths,
            'full_journeys': full_journeys
        }
    
    def find_paths_by_traffic_source(self, start_screen: str, end_screen: str, 
                                    max_length: int = 10, include_context: bool = False,
                                    context_steps: int = 3) -> Dict[str, Dict]:
        """Find paths from start to end, segmented by traffic source.
        
        Args:
            start_screen: Starting screen name
            end_screen: Ending screen name
            max_length: Maximum path length between start and end
            include_context: Whether to include preceding context
            context_steps: Number of preceding steps to include as context
            
        Returns:
            Dictionary with traffic sources as keys, each containing:
                - paths: List of (path, count) tuples
                - total_sessions: Total sessions from this source
                - conversion_rate: Percentage of sessions that complete the path
                - context_paths: (if include_context) Dictionary of context paths
                - full_journeys: (if include_context) Complete journeys with context
        """
        results_by_source = {}
        
        for source in self.traffic_sources:
            source_sessions = self.sessions_by_source.get(source, [])
            
            if not source_sessions:
                continue
            
            # Calculate paths for this traffic source
            path_counts = defaultdict(int)
            context_paths = {}
            journey_counts = defaultdict(int)
            sessions_with_path = 0
            
            for session in source_sessions:
                screens = session['screens']
                session_count = session.get('count', 1)  # Use aggregated count
                found_path = False
                
                for i, screen in enumerate(screens):
                    if screen == start_screen:
                        for j in range(i + 1, min(i + max_length + 1, len(screens))):
                            if screens[j] == end_screen:
                                found_path = True
                                path = screens[i:j+1]
                                path_str = ' -> '.join(path)
                                path_counts[path_str] += session_count
                                
                                if include_context:
                                    preceding = self.get_preceding_steps(screens, start_screen, context_steps)
                                    
                                    if preceding:
                                        context_str = ' -> '.join(preceding)
                                        if path_str not in context_paths:
                                            context_paths[path_str] = defaultdict(int)
                                        context_paths[path_str][context_str] += session_count
                                        
                                        full_journey = preceding + path
                                        journey_str = ' -> '.join(full_journey)
                                        journey_counts[journey_str] += session_count
                                    else:
                                        journey_counts[path_str] += session_count
                                
                                break
                
                if found_path:
                    sessions_with_path += session_count
            
            # Sort and format results
            sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
            paths = [(path.split(' -> '), count) for path, count in sorted_paths]
            
            # Calculate total sessions considering aggregated counts
            total_sessions = sum(session.get('count', 1) for session in source_sessions)
            
            result = {
                'paths': paths,
                'total_sessions': total_sessions,
                'sessions_with_path': sessions_with_path,
                'conversion_rate': (sessions_with_path / total_sessions * 100) if total_sessions else 0
            }
            
            if include_context:
                sorted_journeys = sorted(journey_counts.items(), key=lambda x: x[1], reverse=True)
                result['full_journeys'] = [(journey.split(' -> '), count) for journey, count in sorted_journeys]
                result['context_paths'] = context_paths
            
            results_by_source[source] = result
        
        return results_by_source
    
    def find_paths_by_field(self, start_screen: str, end_screen: str, 
                           segment_field: str, max_length: int = 10, 
                           include_context: bool = False, context_steps: int = 3) -> Dict[str, Dict]:
        """Find paths from start to end, segmented by any field.
        
        Args:
            start_screen: Starting screen name
            end_screen: Ending screen name
            segment_field: Field name to segment by (e.g., 'traffic_source', 'user_type', 'campaign_id')
            max_length: Maximum path length between start and end
            include_context: Whether to include preceding context
            context_steps: Number of preceding steps to include as context
            
        Returns:
            Dictionary with field values as keys, each containing:
                - paths: List of (path, count) tuples
                - total_sessions: Total sessions with this field value
                - conversion_rate: Percentage of sessions that complete the path
                - field_name: The name of the field being segmented
        """
        results_by_value = {}
        
        # Get unique values for this field
        if segment_field in self.sessions_by_segment:
            segment_sessions = self.sessions_by_segment[segment_field]
        else:
            # Fallback to manual grouping if field wasn't pre-indexed
            segment_sessions = defaultdict(list)
            for session in self.sessions:
                value = session.get(segment_field, 'unknown')
                segment_sessions[value].append(session)
        
        for value, value_sessions in segment_sessions.items():
            if not value_sessions:
                continue
            
            # Calculate paths for this segment value
            path_counts = defaultdict(int)
            context_paths = {}
            journey_counts = defaultdict(int)
            sessions_with_path = 0
            
            for session in value_sessions:
                screens = session['screens']
                session_count = session.get('count', 1)  # Use aggregated count
                found_path = False
                
                for i, screen in enumerate(screens):
                    if screen == start_screen:
                        for j in range(i + 1, min(i + max_length + 1, len(screens))):
                            if screens[j] == end_screen:
                                found_path = True
                                path = screens[i:j+1]
                                path_str = ' -> '.join(path)
                                path_counts[path_str] += session_count
                                
                                if include_context:
                                    preceding = self.get_preceding_steps(screens, start_screen, context_steps)
                                    
                                    if preceding:
                                        context_str = ' -> '.join(preceding)
                                        if path_str not in context_paths:
                                            context_paths[path_str] = defaultdict(int)
                                        context_paths[path_str][context_str] += session_count
                                        
                                        full_journey = preceding + path
                                        journey_str = ' -> '.join(full_journey)
                                        journey_counts[journey_str] += session_count
                                    else:
                                        journey_counts[path_str] += session_count
                                
                                break
                
                if found_path:
                    sessions_with_path += session_count
            
            # Sort and format results
            sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
            paths = [(path.split(' -> '), count) for path, count in sorted_paths]
            
            # Calculate total sessions considering aggregated counts
            total_sessions = sum(session.get('count', 1) for session in value_sessions)
            
            result = {
                'paths': paths,
                'total_sessions': total_sessions,
                'sessions_with_path': sessions_with_path,
                'conversion_rate': (sessions_with_path / total_sessions * 100) if total_sessions else 0,
                'field_name': segment_field,
                'field_value': value
            }
            
            if include_context:
                sorted_journeys = sorted(journey_counts.items(), key=lambda x: x[1], reverse=True)
                result['full_journeys'] = [(journey.split(' -> '), count) for journey, count in sorted_journeys]
                result['context_paths'] = context_paths
            
            results_by_value[value] = result
        
        return results_by_value
    
    def find_paths_by_multiple_fields(self, start_screen: str, end_screen: str,
                                     segment_fields: List[str], max_length: int = 10,
                                     include_context: bool = False, context_steps: int = 3) -> Dict:
        """Find paths segmented by multiple fields (creates combinations).
        
        Args:
            start_screen: Starting screen name
            end_screen: Ending screen name
            segment_fields: List of field names to segment by
            max_length: Maximum path length
            
        Returns:
            Dictionary with combined field values as keys
        """
        results_by_combination = {}
        
        # Group sessions by combination of field values
        combination_sessions = defaultdict(list)
        for session in self.sessions:
            # Create combination key from all segment fields
            combo_values = []
            for field in segment_fields:
                value = session.get(field, 'unknown')
                combo_values.append(f"{field}={value}")
            combo_key = " & ".join(combo_values)
            combination_sessions[combo_key].append(session)
        
        # Process each combination
        for combo_key, combo_sessions_list in combination_sessions.items():
            path_counts = defaultdict(int)
            full_journey_counts = defaultdict(int) if include_context else None
            sessions_with_path = 0
            
            for session in combo_sessions_list:
                screens = session['screens']
                session_count = session.get('count', 1)  # Use aggregated count
                found_path = False
                
                for i, screen in enumerate(screens):
                    if screen == start_screen:
                        for j in range(i + 1, min(i + max_length + 1, len(screens))):
                            if screens[j] == end_screen:
                                found_path = True
                                path = screens[i:j+1]
                                path_str = ' -> '.join(path)
                                path_counts[path_str] += session_count
                                
                                # Capture context if requested
                                if include_context:
                                    context_start = max(0, i - context_steps)
                                    full_journey = screens[context_start:j+1]
                                    full_journey_str = ' -> '.join(full_journey)
                                    full_journey_counts[full_journey_str] += session_count
                                break
                
                if found_path:
                    sessions_with_path += session_count
            
            sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
            paths = [(path.split(' -> '), count) for path, count in sorted_paths]
            
            # Calculate total sessions considering aggregated counts
            total_sessions = sum(session.get('count', 1) for session in combo_sessions_list)
            
            result = {
                'paths': paths,
                'total_sessions': total_sessions,
                'sessions_with_path': sessions_with_path,
                'conversion_rate': (sessions_with_path / total_sessions * 100) if total_sessions else 0,
                'fields': segment_fields,
                'combination': combo_key
            }
            
            if include_context and full_journey_counts:
                sorted_journeys = sorted(full_journey_counts.items(), key=lambda x: x[1], reverse=True)
                result['full_journeys'] = [(journey.split(' -> '), count) for journey, count in sorted_journeys]
            
            results_by_combination[combo_key] = result
        
        return results_by_combination
    
    def get_available_fields(self) -> List[str]:
        """Get list of all available fields that can be used for segmentation."""
        return list(self.segment_values.keys())
    
    def get_field_values(self, field_name: str) -> List[str]:
        """Get all unique values for a specific field."""
        return list(self.segment_values.get(field_name, set()))
    
    def get_statistics(self) -> Dict:
        # Calculate total sessions considering aggregated counts
        total_sessions = sum(session.get('count', 1) for session in self.sessions) if self.is_pre_aggregated else len(self.sessions)
        
        # Calculate weighted average path length for aggregated data
        if self.is_pre_aggregated:
            total_length = sum(len(s['screens']) * s.get('count', 1) for s in self.sessions)
            avg_path_length = total_length / total_sessions if total_sessions else 0
        else:
            avg_path_length = np.mean([len(s['screens']) for s in self.sessions]) if self.sessions else 0
        
        all_screens = set()
        screen_visits = Counter()
        
        for session in self.sessions:
            all_screens.update(session['screens'])
            session_count = session.get('count', 1)
            for screen in session['screens']:
                screen_visits[screen] += session_count
        
        return {
            'total_sessions': total_sessions,
            'unique_screens': len(all_screens),
            'average_path_length': round(avg_path_length, 2),
            'most_visited_screens': screen_visits.most_common(10),
            'total_transitions': sum(sum(transitions.values()) 
                                   for transitions in self.screen_transitions.values())
        }
    
    def find_most_common_paths_to_screen(self, end_screen: str, max_length: int = 10, 
                                        top_n: int = 10) -> List[Tuple[List[str], int]]:
        """Find the most common paths that lead to a specific end screen.
        
        Args:
            end_screen: The target screen to find paths to
            max_length: Maximum path length to consider
            top_n: Number of top paths to return
            
        Returns:
            List of (path, count) tuples sorted by frequency
        """
        path_counts = defaultdict(int)
        
        for session in self.sessions:
            screens = session['screens']
            session_count = session.get('count', 1)  # Use aggregated count
            
            # Find all occurrences of end_screen in this session
            for j, screen in enumerate(screens):
                if screen == end_screen:
                    # Look backwards to find possible starting points
                    for i in range(max(0, j - max_length + 1), j):
                        path = screens[i:j+1]
                        if len(path) >= 2:  # At least start->end
                            path_str = ' -> '.join(path)
                            path_counts[path_str] += session_count
        
        # Sort by frequency and return top results
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(path.split(' -> '), count) for path, count in sorted_paths]
    
    def find_least_common_paths_to_screen(self, end_screen: str, max_length: int = 10, 
                                         top_n: int = 10) -> List[Tuple[List[str], int]]:
        """Find the least common paths that lead to a specific end screen.
        
        Args:
            end_screen: The target screen to find paths to
            max_length: Maximum path length to consider
            top_n: Number of least common paths to return
            
        Returns:
            List of (path, count) tuples sorted by frequency (ascending)
        """
        path_counts = defaultdict(int)
        
        for session in self.sessions:
            screens = session['screens']
            session_count = session.get('count', 1)  # Use aggregated count
            
            # Find all occurrences of end_screen in this session
            for j, screen in enumerate(screens):
                if screen == end_screen:
                    # Look backwards to find paths leading to this screen
                    start_idx = max(0, j - max_length + 1)
                    path = screens[start_idx:j+1]
                    if len(path) >= 2:  # At least start->end
                        path_str = ' -> '.join(path)
                        path_counts[path_str] += session_count
        
        # Sort by frequency (ascending) and return bottom results
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1])[:top_n]
        return [(path.split(' -> '), count) for path, count in sorted_paths]
    
    def find_least_common_paths_to_screen_with_filter(self, end_screen: str, 
                                                     field_filter: Dict[str, str] = None,
                                                     max_length: int = 10, 
                                                     top_n: int = 10) -> List[Tuple[List[str], int]]:
        """Find the least common paths that lead to a specific end screen with field filtering.
        
        Args:
            end_screen: The target screen to find paths to
            field_filter: Dictionary of field->value filters to apply to target screen events
            max_length: Maximum path length to consider
            top_n: Number of least common paths to return
            
        Returns:
            List of (path, count) tuples sorted by frequency (ascending)
        """
        path_counts = defaultdict(int)
        
        # For field filtering, we need to access the original event data
        if field_filter and not self.is_pre_aggregated:
            # This is complex for raw event data, so let's implement a basic version
            # that filters sessions by metadata fields
            filtered_sessions = []
            for session in self.sessions:
                matches = True
                for field, expected_value in field_filter.items():
                    if field in session:
                        session_value = session[field]
                        if str(session_value).lower() != str(expected_value).lower():
                            matches = False
                            break
                if matches:
                    filtered_sessions.append(session)
        elif field_filter and self.is_pre_aggregated:
            # For pre-aggregated data, filter by chunk metadata
            filtered_sessions = []
            for session in self.sessions:
                matches = True
                for field, expected_value in field_filter.items():
                    if field in session:
                        session_value = session[field]
                        if str(session_value).lower() != str(expected_value).lower():
                            matches = False
                            break
                if matches:
                    filtered_sessions.append(session)
        else:
            filtered_sessions = self.sessions
        
        for session in filtered_sessions:
            screens = session['screens']
            session_count = session.get('count', 1)
            
            # Find all occurrences of end_screen in this session
            for j, screen in enumerate(screens):
                if screen == end_screen:
                    # Look backwards to find paths leading to this screen
                    start_idx = max(0, j - max_length + 1)
                    path = screens[start_idx:j+1]
                    if len(path) >= 2:  # At least start->end
                        path_str = ' -> '.join(path)
                        path_counts[path_str] += session_count
        
        # Sort by frequency (ascending) and return bottom results
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1])[:top_n]
        return [(path.split(' -> '), count) for path, count in sorted_paths]
    
    def find_most_common_paths_to_screen_with_filter(self, end_screen: str, 
                                                    field_filter: Dict[str, str] = None,
                                                    max_length: int = 10, 
                                                    top_n: int = 10) -> List[Tuple[List[str], int]]:
        """Find the most common paths that lead to a specific end screen with field filtering.
        
        Args:
            end_screen: The target screen to find paths to
            field_filter: Dictionary of field->value filters to apply to target screen events
            max_length: Maximum path length to consider
            top_n: Number of top paths to return
            
        Returns:
            List of (path, count) tuples sorted by frequency
        """
        path_counts = defaultdict(int)
        
        # For field filtering, we need to access the original event data
        if field_filter and not self.is_pre_aggregated:
            # Need to rebuild the data structure to include event-level metadata
            # This is more complex for raw event data, so let's implement it differently
            return self._find_paths_with_event_filter(end_screen, field_filter, max_length, top_n)
        
        for session in self.sessions:
            screens = session['screens']
            
            # Find all occurrences of end_screen in this session
            for j, screen in enumerate(screens):
                if screen == end_screen:
                    # Check if this screen event matches the filter criteria
                    if field_filter:
                        # For pre-aggregated data, check if session metadata matches filter
                        filter_match = True
                        for field, expected_value in field_filter.items():
                            session_value = session.get(field)
                            if session_value is None or str(session_value).lower() != str(expected_value).lower():
                                filter_match = False
                                break
                        if not filter_match:
                            continue
                    
                    # Look backwards to find possible starting points
                    for i in range(max(0, j - max_length + 1), j):
                        path = screens[i:j+1]
                        if len(path) >= 2:  # At least start->end
                            path_str = ' -> '.join(path)
                            # Apply the count multiplier if available (for pre-aggregated data)
                            count_multiplier = session.get('count', 1)
                            path_counts[path_str] += count_multiplier
        
        # Sort by frequency and return top results
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(path.split(' -> '), count) for path, count in sorted_paths]
    
    def _find_paths_with_event_filter(self, end_screen: str, field_filter: Dict[str, str], 
                                     max_length: int, top_n: int) -> List[Tuple[List[str], int]]:
        """Helper method to find paths with event-level filtering for raw data."""
        path_counts = defaultdict(int)
        
        for session in self.sessions:
            screens = session['screens']
            session_id = session['session_id']
            
            # Find all occurrences of end_screen in this session
            for j, screen in enumerate(screens):
                if screen == end_screen:
                    # Check event-level metadata for this specific screen occurrence
                    event_meta = self.event_metadata.get(session_id, {}).get(j, {})
                    
                    # Apply filter to event-level metadata
                    filter_match = True
                    for field, expected_value in field_filter.items():
                        event_value = event_meta.get(field)
                        if event_value is None:
                            # Fall back to session-level metadata if event-level not available
                            event_value = session.get(field)
                        
                        if event_value is None or str(event_value).lower() != str(expected_value).lower():
                            filter_match = False
                            break
                    
                    if not filter_match:
                        continue
                    
                    # Look backwards to find possible starting points
                    for i in range(max(0, j - max_length + 1), j):
                        path = screens[i:j+1]
                        if len(path) >= 2:  # At least start->end
                            path_str = ' -> '.join(path)
                            path_counts[path_str] += 1
        
        # Sort by frequency and return top results
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(path.split(' -> '), count) for path, count in sorted_paths]

    def export_transition_matrix(self) -> pd.DataFrame:
        screens = sorted(set(list(self.screen_transitions.keys()) + 
                           [s for transitions in self.screen_transitions.values() 
                            for s in transitions.keys()]))
        
        matrix = pd.DataFrame(0, index=screens, columns=screens)
        
        for from_screen, transitions in self.screen_transitions.items():
            for to_screen, count in transitions.items():
                matrix.loc[from_screen, to_screen] = count
        
        return matrix
    
    def export_transition_probability_matrix(self) -> pd.DataFrame:
        """Export transition matrix with probabilities instead of raw counts.
        
        For each 'from' screen, shows the probability of transitioning to each 'to' screen.
        Probabilities sum to 1.0 for each row (from screen).
        
        Returns:
            pd.DataFrame: Matrix where each cell contains the probability of transition
        """
        screens = sorted(set(list(self.screen_transitions.keys()) + 
                           [s for transitions in self.screen_transitions.values() 
                            for s in transitions.keys()]))
        
        matrix = pd.DataFrame(0.0, index=screens, columns=screens)
        
        for from_screen, transitions in self.screen_transitions.items():
            # Calculate total transitions from this screen
            total_from_screen = sum(transitions.values())
            
            if total_from_screen > 0:
                for to_screen, count in transitions.items():
                    # Calculate probability as count / total_from_screen
                    probability = count / total_from_screen
                    matrix.loc[from_screen, to_screen] = probability
        
        return matrix