#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import random
from flow_analyzer import FlowPathAnalyzer


class MonteCarloSimulator:
    def __init__(self, analyzer: FlowPathAnalyzer):
        """Initialize Monte Carlo simulator with flow analyzer data.
        
        Args:
            analyzer: FlowPathAnalyzer instance with loaded data
        """
        self.analyzer = analyzer
        self.baseline_probabilities = analyzer.export_transition_probability_matrix()
        self.modified_probabilities = self.baseline_probabilities.copy()
        self.baseline_metrics = self._calculate_baseline_metrics()
        
    def _calculate_baseline_metrics(self) -> Dict:
        """Calculate baseline metrics from current data."""
        stats = self.analyzer.get_statistics()
        
        # Calculate conversion rates for key paths
        conversion_paths = {}
        common_goals = ['OrderConfirmation', 'Checkout', 'Payment', 'Purchase']
        
        for goal in common_goals:
            if goal in self.baseline_probabilities.columns:
                # Find paths that lead to this goal
                goal_sessions = sum(1 for session in self.analyzer.sessions 
                                  if goal in session['screens'])
                total_sessions = len(self.analyzer.sessions)
                conversion_paths[goal] = (goal_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        return {
            'total_sessions': stats['total_sessions'],
            'average_path_length': stats['average_path_length'],
            'conversion_rates': conversion_paths,
            'unique_screens': stats['unique_screens']
        }
    
    def simulate_user_journey(self, start_screen: str = None, 
                            max_steps: int = 20, 
                            use_modified: bool = True) -> List[str]:
        """Simulate a single user journey using transition probabilities.
        
        Args:
            start_screen: Starting screen (random if None)
            max_steps: Maximum number of steps in journey
            use_modified: Use modified probabilities vs baseline
            
        Returns:
            List of screens representing the user journey
        """
        prob_matrix = self.modified_probabilities if use_modified else self.baseline_probabilities
        
        if not start_screen:
            # Choose random starting screen weighted by how often it appears as first screen
            first_screens = [session['screens'][0] for session in self.analyzer.sessions 
                           if len(session['screens']) > 0]
            start_screen = random.choice(first_screens) if first_screens else random.choice(prob_matrix.index.tolist())
        
        journey = [start_screen]
        current_screen = start_screen
        
        for _ in range(max_steps):
            if current_screen not in prob_matrix.index:
                break  # No transitions available from this screen
            
            # Get transition probabilities for current screen
            transitions = prob_matrix.loc[current_screen]
            transitions = transitions[transitions > 0]  # Only non-zero probabilities
            
            if len(transitions) == 0:
                break  # No valid transitions (exit point)
            
            # Choose next screen based on probabilities
            next_screens = transitions.index.tolist()
            probabilities = transitions.values.tolist()
            
            # Add small probability of exit (user leaves the site)
            exit_probability = 0.1
            next_screens.append('__EXIT__')
            probabilities.append(exit_probability)
            
            # Normalize probabilities
            prob_sum = sum(probabilities)
            probabilities = [p / prob_sum for p in probabilities]
            
            next_screen = np.random.choice(next_screens, p=probabilities)
            
            if next_screen == '__EXIT__':
                break
                
            journey.append(next_screen)
            current_screen = next_screen
        
        return journey
    
    def run_simulation(self, n_simulations: int = 1000, 
                      start_screen: str = None,
                      use_modified: bool = True,
                      goal_screens: List[str] = None) -> Dict:
        """Run Monte Carlo simulation with multiple user journeys.
        
        Args:
            n_simulations: Number of user journeys to simulate
            start_screen: Starting screen for all journeys (random if None)
            use_modified: Use modified probabilities vs baseline
            goal_screens: List of screens to track as conversion goals (auto-detect if None)
            
        Returns:
            Dictionary with simulation results and metrics
        """
        journeys = []
        path_counts = defaultdict(int)
        screen_visits = Counter()
        conversion_counts = defaultdict(int)
        
        for _ in range(n_simulations):
            journey = self.simulate_user_journey(start_screen, use_modified=use_modified)
            journeys.append(journey)
            
            # Count screen visits
            screen_visits.update(journey)
            
            # Count conversions to goal screens
            goals_to_track = goal_screens if goal_screens else self.baseline_metrics['conversion_rates'].keys()
            for goal in goals_to_track:
                if goal in journey:
                    conversion_counts[goal] += 1
            
            # Count path patterns of various lengths, avoiding repetitive loops
            max_path_length = min(len(journey), 6)  # Up to 5-step paths
            for path_length in range(2, max_path_length):  # 2 to 5 steps
                for i in range(len(journey) - path_length + 1):
                    path_segment = journey[i:i + path_length]
                    
                    # Skip paths that are mostly repetitive loops (same screen >50% of path)
                    unique_screens = set(path_segment)
                    if len(unique_screens) / len(path_segment) < 0.5:
                        continue
                    
                    path = " -> ".join(path_segment)
                    path_counts[path] += 1
        
        # Calculate metrics
        avg_journey_length = np.mean([len(j) for j in journeys])
        conversion_rates = {goal: (count / n_simulations * 100) 
                          for goal, count in conversion_counts.items()}
        
        # Most common paths
        top_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        results = {
            'journeys': journeys,
            'n_simulations': n_simulations,
            'avg_journey_length': avg_journey_length,
            'conversion_rates': conversion_rates,
            'screen_visits': dict(screen_visits),
            'top_paths': top_paths,
            'total_screen_visits': sum(screen_visits.values())
        }
        
        return results
    
    def modify_transition(self, from_screen: str, to_screen: str, 
                         new_probability: float) -> bool:
        """Modify transition probability between two screens.
        
        Args:
            from_screen: Source screen
            to_screen: Destination screen  
            new_probability: New probability (0.0 to 1.0)
            
        Returns:
            True if modification successful, False otherwise
        """
        if from_screen not in self.modified_probabilities.index:
            print(f"Screen '{from_screen}' not found in transition matrix")
            return False
        
        if to_screen not in self.modified_probabilities.columns:
            print(f"Screen '{to_screen}' not found in transition matrix")
            return False
        
        if not 0.0 <= new_probability <= 1.0:
            print("Probability must be between 0.0 and 1.0")
            return False
        
        # Store original probability
        original_prob = self.modified_probabilities.loc[from_screen, to_screen]
        
        # Set new probability
        self.modified_probabilities.loc[from_screen, to_screen] = new_probability
        
        # Normalize row to ensure probabilities sum to ≤ 1.0
        row_sum = self.modified_probabilities.loc[from_screen].sum()
        if row_sum > 1.0:
            # Scale down other probabilities proportionally
            other_screens = [col for col in self.modified_probabilities.columns if col != to_screen]
            remaining_prob = 1.0 - new_probability
            current_other_sum = self.modified_probabilities.loc[from_screen, other_screens].sum()
            
            if current_other_sum > 0:
                scale_factor = remaining_prob / current_other_sum
                self.modified_probabilities.loc[from_screen, other_screens] *= scale_factor
        
        print(f"Modified transition {from_screen} -> {to_screen}: {original_prob:.3f} -> {new_probability:.3f}")
        return True
    
    def increase_conversion(self, from_screen: str, to_screen: str, 
                          increase_percent: float) -> bool:
        """Increase transition probability by a percentage.
        
        Args:
            from_screen: Source screen
            to_screen: Destination screen
            increase_percent: Percentage increase (e.g., 25.0 for 25% increase)
            
        Returns:
            True if modification successful, False otherwise
        """
        if from_screen not in self.modified_probabilities.index:
            print(f"Screen '{from_screen}' not found")
            return False
            
        current_prob = self.modified_probabilities.loc[from_screen, to_screen]
        new_prob = current_prob * (1 + increase_percent / 100)
        new_prob = min(new_prob, 1.0)  # Cap at 100%
        
        return self.modify_transition(from_screen, to_screen, new_prob)
    
    def compare_scenarios(self, n_simulations: int = 1000, 
                         start_screen: str = None,
                         goal_screens: List[str] = None) -> Dict:
        """Compare baseline vs modified scenario simulation results.
        
        Args:
            n_simulations: Number of simulations to run for each scenario
            start_screen: Starting screen for simulations
            goal_screens: List of screens to track as conversion goals
            
        Returns:
            Dictionary comparing baseline and modified results
        """
        print(f"Running baseline simulation ({n_simulations:,} journeys)...")
        baseline_results = self.run_simulation(n_simulations, start_screen, use_modified=False, goal_screens=goal_screens)
        
        print(f"Running modified simulation ({n_simulations:,} journeys)...")
        modified_results = self.run_simulation(n_simulations, start_screen, use_modified=True, goal_screens=goal_screens)
        
        # Calculate differences
        conversion_diff = {}
        for goal in baseline_results['conversion_rates'].keys():
            baseline_rate = baseline_results['conversion_rates'][goal]
            modified_rate = modified_results['conversion_rates'].get(goal, 0)
            diff = modified_rate - baseline_rate
            percent_change = (diff / baseline_rate * 100) if baseline_rate > 0 else 0
            conversion_diff[goal] = {
                'baseline': baseline_rate,
                'modified': modified_rate,
                'absolute_change': diff,
                'percent_change': percent_change
            }
        
        journey_length_diff = modified_results['avg_journey_length'] - baseline_results['avg_journey_length']
        journey_length_percent = (journey_length_diff / baseline_results['avg_journey_length'] * 100) if baseline_results['avg_journey_length'] > 0 else 0
        
        return {
            'baseline': baseline_results,
            'modified': modified_results,
            'conversion_changes': conversion_diff,
            'journey_length_change': {
                'absolute': journey_length_diff,
                'percent': journey_length_percent
            },
            'n_simulations': n_simulations
        }
    
    def reset_modifications(self):
        """Reset all modifications to baseline probabilities."""
        self.modified_probabilities = self.baseline_probabilities.copy()
        print("Reset to baseline probabilities")
    
    def get_modification_summary(self) -> List[Dict]:
        """Get summary of all current modifications.
        
        Returns:
            List of dictionaries describing each modification
        """
        modifications = []
        
        for from_screen in self.modified_probabilities.index:
            for to_screen in self.modified_probabilities.columns:
                baseline_prob = self.baseline_probabilities.loc[from_screen, to_screen]
                modified_prob = self.modified_probabilities.loc[from_screen, to_screen]
                
                if abs(baseline_prob - modified_prob) > 0.001:  # Significant difference
                    change = modified_prob - baseline_prob
                    percent_change = (change / baseline_prob * 100) if baseline_prob > 0 else float('inf')
                    
                    modifications.append({
                        'from_screen': from_screen,
                        'to_screen': to_screen,
                        'baseline_probability': baseline_prob,
                        'modified_probability': modified_prob,
                        'absolute_change': change,
                        'percent_change': percent_change
                    })
        
        return modifications