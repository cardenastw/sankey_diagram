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
        
        for event in data:
            session_id = event.get('session_id', event.get('user_id', 'default'))
            screen = event.get('screen', event.get('screen_view', event.get('page')))
            timestamp = event.get('timestamp', None)
            
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
                self.sessions.append({
                    'session_id': session_id,
                    'screens': screen_sequence
                })
                
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