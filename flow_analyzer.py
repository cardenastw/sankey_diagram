import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
import json


class FlowPathAnalyzer:
    def __init__(self):
        self.sessions = []
        self.flow_paths = defaultdict(int)
        self.screen_transitions = defaultdict(lambda: defaultdict(int))
        self.sessions_by_source = defaultdict(list)
        self.traffic_sources = set()
        self.sessions_by_segment = defaultdict(lambda: defaultdict(list))
        self.segment_values = defaultdict(set)
        
    def load_data(self, data_source):
        if isinstance(data_source, str):
            if data_source.endswith('.json'):
                with open(data_source, 'r') as f:
                    data = json.load(f)
            elif data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
                data = df.to_dict('records')
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV.")
        elif isinstance(data_source, list):
            data = data_source
        else:
            raise ValueError("Data source must be a file path or list of dictionaries")
        
        self._process_sessions(data)
        
    def _process_sessions(self, data):
        sessions_dict = defaultdict(list)
        session_sources = {}
        session_metadata = defaultdict(dict)
        
        for event in data:
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
                sessions_dict[session_id].append({
                    'screen': screen,
                    'timestamp': timestamp
                })
        
        for session_id, screens in sessions_dict.items():
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
    
    def find_paths(self, start_screen: str, end_screen: str, max_length: int = 10) -> List[Tuple[List[str], int]]:
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
        all_paths = defaultdict(int)
        
        for session in self.sessions:
            screens = session['screens']
            for length in range(2, min(6, len(screens) + 1)):
                for i in range(len(screens) - length + 1):
                    path = tuple(screens[i:i+length])
                    all_paths[path] += 1
        
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
            
            for i, screen in enumerate(screens):
                if screen == start_screen:
                    # Find if this start leads to the end
                    for j in range(i + 1, min(i + max_length + 1, len(screens))):
                        if screens[j] == end_screen:
                            # Found a path from start to end
                            path = screens[i:j+1]
                            path_str = ' -> '.join(path)
                            path_counts[path_str] += 1
                            
                            # Get preceding context
                            preceding = self.get_preceding_steps(screens, start_screen, context_steps)
                            
                            if preceding:
                                context_str = ' -> '.join(preceding)
                                if path_str not in context_paths:
                                    context_paths[path_str] = defaultdict(int)
                                context_paths[path_str][context_str] += 1
                                
                                # Create full journey
                                full_journey = preceding + path
                                journey_str = ' -> '.join(full_journey)
                                journey_counts[journey_str] += 1
                            else:
                                # No preceding steps, just the path itself
                                journey_str = path_str
                                journey_counts[journey_str] += 1
                            
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
                found_path = False
                
                for i, screen in enumerate(screens):
                    if screen == start_screen:
                        for j in range(i + 1, min(i + max_length + 1, len(screens))):
                            if screens[j] == end_screen:
                                found_path = True
                                path = screens[i:j+1]
                                path_str = ' -> '.join(path)
                                path_counts[path_str] += 1
                                
                                if include_context:
                                    preceding = self.get_preceding_steps(screens, start_screen, context_steps)
                                    
                                    if preceding:
                                        context_str = ' -> '.join(preceding)
                                        if path_str not in context_paths:
                                            context_paths[path_str] = defaultdict(int)
                                        context_paths[path_str][context_str] += 1
                                        
                                        full_journey = preceding + path
                                        journey_str = ' -> '.join(full_journey)
                                        journey_counts[journey_str] += 1
                                    else:
                                        journey_counts[path_str] += 1
                                
                                break
                
                if found_path:
                    sessions_with_path += 1
            
            # Sort and format results
            sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
            paths = [(path.split(' -> '), count) for path, count in sorted_paths]
            
            result = {
                'paths': paths,
                'total_sessions': len(source_sessions),
                'sessions_with_path': sessions_with_path,
                'conversion_rate': (sessions_with_path / len(source_sessions) * 100) if source_sessions else 0
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
                found_path = False
                
                for i, screen in enumerate(screens):
                    if screen == start_screen:
                        for j in range(i + 1, min(i + max_length + 1, len(screens))):
                            if screens[j] == end_screen:
                                found_path = True
                                path = screens[i:j+1]
                                path_str = ' -> '.join(path)
                                path_counts[path_str] += 1
                                
                                if include_context:
                                    preceding = self.get_preceding_steps(screens, start_screen, context_steps)
                                    
                                    if preceding:
                                        context_str = ' -> '.join(preceding)
                                        if path_str not in context_paths:
                                            context_paths[path_str] = defaultdict(int)
                                        context_paths[path_str][context_str] += 1
                                        
                                        full_journey = preceding + path
                                        journey_str = ' -> '.join(full_journey)
                                        journey_counts[journey_str] += 1
                                    else:
                                        journey_counts[path_str] += 1
                                
                                break
                
                if found_path:
                    sessions_with_path += 1
            
            # Sort and format results
            sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
            paths = [(path.split(' -> '), count) for path, count in sorted_paths]
            
            result = {
                'paths': paths,
                'total_sessions': len(value_sessions),
                'sessions_with_path': sessions_with_path,
                'conversion_rate': (sessions_with_path / len(value_sessions) * 100) if value_sessions else 0,
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
                found_path = False
                
                for i, screen in enumerate(screens):
                    if screen == start_screen:
                        for j in range(i + 1, min(i + max_length + 1, len(screens))):
                            if screens[j] == end_screen:
                                found_path = True
                                path = screens[i:j+1]
                                path_str = ' -> '.join(path)
                                path_counts[path_str] += 1
                                
                                # Capture context if requested
                                if include_context:
                                    context_start = max(0, i - context_steps)
                                    full_journey = screens[context_start:j+1]
                                    full_journey_str = ' -> '.join(full_journey)
                                    full_journey_counts[full_journey_str] += 1
                                break
                
                if found_path:
                    sessions_with_path += 1
            
            sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
            paths = [(path.split(' -> '), count) for path, count in sorted_paths]
            
            result = {
                'paths': paths,
                'total_sessions': len(combo_sessions_list),
                'sessions_with_path': sessions_with_path,
                'conversion_rate': (sessions_with_path / len(combo_sessions_list) * 100) if combo_sessions_list else 0,
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
        total_sessions = len(self.sessions)
        avg_path_length = np.mean([len(s['screens']) for s in self.sessions]) if self.sessions else 0
        
        all_screens = set()
        for session in self.sessions:
            all_screens.update(session['screens'])
        
        screen_visits = Counter()
        for session in self.sessions:
            screen_visits.update(session['screens'])
        
        return {
            'total_sessions': total_sessions,
            'unique_screens': len(all_screens),
            'average_path_length': round(avg_path_length, 2),
            'most_visited_screens': screen_visits.most_common(10),
            'total_transitions': sum(sum(transitions.values()) 
                                   for transitions in self.screen_transitions.values())
        }
    
    def export_transition_matrix(self) -> pd.DataFrame:
        screens = sorted(set(list(self.screen_transitions.keys()) + 
                           [s for transitions in self.screen_transitions.values() 
                            for s in transitions.keys()]))
        
        matrix = pd.DataFrame(0, index=screens, columns=screens)
        
        for from_screen, transitions in self.screen_transitions.items():
            for to_screen, count in transitions.items():
                matrix.loc[from_screen, to_screen] = count
        
        return matrix