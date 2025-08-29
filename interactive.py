#!/usr/bin/env python3

import argparse
from flow_analyzer import FlowPathAnalyzer
from monte_carlo import MonteCarloSimulator
from visualizer import FlowPathVisualizer


class InteractiveFlowAnalyzer:
    def __init__(self, data_source: str, is_pre_aggregated: bool = False, show_progress: bool = True):
        self.show_progress = show_progress
        
        # Initialize analyzer with progress tracking
        if show_progress:
            print("Initializing analyzer...")
        self.analyzer = FlowPathAnalyzer(is_pre_aggregated=is_pre_aggregated, show_progress=show_progress)
        
        # Load data with progress updates
        if show_progress:
            print(f"Loading data from {data_source}...")
        self.analyzer.load_data(data_source)
        if show_progress:
            print("Data loaded successfully!")
        
        # Initialize Monte Carlo simulator and visualizer
        if show_progress:
            print("Initializing Monte Carlo simulator...")
        self.simulator = MonteCarloSimulator(self.analyzer)
        if show_progress:
            print("Monte Carlo simulator initialized!")
        
        if show_progress:
            print("Initializing visualizer...")
        self.visualizer = FlowPathVisualizer(self.analyzer)
        if show_progress:
            print("Visualizer initialized!")
        
    def show_help(self):
        print("""
Available commands:
  most-common-to <screen> [--where field=value]  - Find most common paths to a screen with optional filtering
  paths <start> <end>                           - Find paths from start to end screen
  stats                                         - Show flow statistics
  fields                                        - List available segmentation fields
  simulate --change <from>-><to> <probability> [--goal screen1,screen2] - Modify transition and run simulation
  simulate --increase <from>-><to> <percent> [--goal screen1,screen2]   - Increase transition and run simulation  
  simulate --compare [n_simulations] [--goal screen1,screen2]           - Compare baseline vs modified scenarios
  simulate --visualize comparison [--goal screen1,screen2]              - Create comparison chart
  simulate --visualize sankey <from>-><to>                             - Create modified Sankey diagram
  simulate --reset                                                     - Reset all modifications to baseline
  simulate --status                                                    - Show current modifications
  help                                          - Show this help message
  exit                                          - Exit the interactive session

Examples:
  most-common-to Cart
  most-common-to Checkout --where purchased=true
  most-common-to Products --where device=mobile
  paths Home Checkout
  stats
  simulate --change Landing->Products 0.4
  simulate --increase Cart->Checkout 25 --goal OrderConfirmation
  simulate --compare 5000 --goal ProductDetail,Reviews
  simulate --visualize comparison --goal Registration,ContactForm
  simulate --visualize sankey Landing->Checkout
        """)
    
    def handle_most_common_to(self, screen: str, top_n: int = 5, field_filter: dict = None):
        if field_filter:
            paths = self.analyzer.find_most_common_paths_to_screen_with_filter(
                screen, field_filter=field_filter, top_n=top_n
            )
            filter_desc = ", ".join([f"{k}={v}" for k, v in field_filter.items()])
            print(f"\nTop {len(paths)} paths leading to '{screen}' where {filter_desc}:")
        else:
            paths = self.analyzer.find_most_common_paths_to_screen(screen, top_n=top_n)
            print(f"\nTop {len(paths)} paths leading to '{screen}':")
        
        if not paths:
            filter_msg = f" with filter {field_filter}" if field_filter else ""
            print(f"No paths found leading to '{screen}'{filter_msg}")
            return
        
        for i, (path, count) in enumerate(paths, 1):
            print(f"{i}. {' -> '.join(path)} (Count: {count})")
    
    def _parse_where_clause(self, parts: list) -> dict:
        """Parse --where field=value clauses from command parts."""
        field_filter = {}
        i = 0
        while i < len(parts):
            if parts[i] == '--where' and i + 1 < len(parts):
                # Parse field=value
                filter_expr = parts[i + 1]
                if '=' in filter_expr:
                    field, value = filter_expr.split('=', 1)
                    # Convert string boolean values
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    field_filter[field] = value
                i += 2
            else:
                i += 1
        return field_filter
    
    def _parse_goal_screens(self, parts: list) -> list:
        """Parse --goal screen1,screen2 from command parts."""
        goal_screens = None
        i = 0
        while i < len(parts):
            if parts[i] == '--goal' and i + 1 < len(parts):
                # Parse comma-separated goals
                goals_str = parts[i + 1]
                goal_screens = [goal.strip() for goal in goals_str.split(',')]
                break
            i += 1
        return goal_screens
    
    def handle_paths(self, start: str, end: str, top_n: int = 5):
        paths = self.analyzer.find_paths(start, end)
        
        if not paths:
            print(f"No paths found from '{start}' to '{end}'")
            return
        
        print(f"\nTop {min(top_n, len(paths))} paths from '{start}' to '{end}':")
        for i, (path, count) in enumerate(paths[:top_n], 1):
            print(f"{i}. {' -> '.join(path)} (Count: {count})")
    
    def handle_stats(self):
        stats = self.analyzer.get_statistics()
        print(f"\nTotal Sessions: {stats['total_sessions']}")
        print(f"Unique Screens: {stats['unique_screens']}")
        print(f"Average Path Length: {stats['average_path_length']}")
        print(f"Total Transitions: {stats['total_transitions']}")
        print("\nMost Visited Screens:")
        for screen, count in stats['most_visited_screens'][:5]:
            print(f"  - {screen}: {count} visits")
    
    def handle_fields(self):
        fields = self.analyzer.get_available_fields()
        if fields:
            print("\nAvailable fields for segmentation:")
            for field in fields:
                values = self.analyzer.get_field_values(field)
                print(f"  - {field}: {len(values)} unique values")
                if len(values) <= 5:
                    print(f"    Values: {', '.join(map(str, values))}")
        else:
            print("\nNo segmentation fields found in the data.")
    
    def handle_simulate(self, parts: list):
        """Handle simulation commands."""
        if len(parts) < 2:
            print("Usage: simulate --change <from>-><to> <prob> | --increase <from>-><to> <percent> | --compare | --reset | --status")
            return
        
        # Parse goal screens if specified
        goal_screens = self._parse_goal_screens(parts)
        if goal_screens:
            print(f"Tracking goals: {', '.join(goal_screens)}")
        
        action = parts[1]
        
        if action == '--change':
            if len(parts) < 4:
                print("Usage: simulate --change <from>-><to> <probability>")
                return
            
            transition = parts[2]
            try:
                probability = float(parts[3])
            except ValueError:
                print("Probability must be a number between 0 and 1")
                return
            
            if '->' not in transition:
                print("Transition must be in format: FromScreen->ToScreen")
                return
            
            from_screen, to_screen = transition.split('->', 1)
            if self.simulator.modify_transition(from_screen, to_screen, probability):
                print(f"\nRunning simulation with modified transition...")
                self._run_quick_simulation(goal_screens=goal_screens)
        
        elif action == '--increase':
            if len(parts) < 4:
                print("Usage: simulate --increase <from>-><to> <percent>")
                return
            
            transition = parts[2]
            try:
                percent = float(parts[3])
            except ValueError:
                print("Percent must be a number")
                return
            
            if '->' not in transition:
                print("Transition must be in format: FromScreen->ToScreen")
                return
            
            from_screen, to_screen = transition.split('->', 1)
            if self.simulator.increase_conversion(from_screen, to_screen, percent):
                print(f"\nRunning simulation with increased conversion...")
                self._run_quick_simulation(goal_screens=goal_screens)
        
        elif action == '--compare':
            n_sims = 1000
            if len(parts) > 2:
                try:
                    n_sims = int(parts[2])
                except ValueError:
                    print("Number of simulations must be an integer")
                    return
            
            self._run_comparison(n_sims, goal_screens=goal_screens)
        
        elif action == '--reset':
            self.simulator.reset_modifications()
        
        elif action == '--status':
            self._show_modifications()
        
        elif action == '--visualize':
            if len(parts) < 3:
                print("Usage: simulate --visualize comparison | sankey <from>-><to>")
                return
            
            viz_type = parts[2]
            if viz_type == 'comparison':
                # Run comparison and create visualization
                n_sims = 1000
                comparison = self.simulator.compare_scenarios(n_sims, goal_screens=goal_screens)
                output_file = f"simulation_comparison_{n_sims}.html"
                self.visualizer.create_simulation_comparison_chart(comparison, output_file)
            
            elif viz_type.startswith('sankey') and len(parts) > 3:
                # Extract start->end from next parameter
                transition = parts[3]
                if '->' in transition:
                    start_screen, end_screen = transition.split('->', 1)
                    output_file = f"modified_sankey_{start_screen}_{end_screen}.html"
                    self.visualizer.create_modified_sankey(
                        self.simulator, start_screen, end_screen, 
                        output_file=output_file
                    )
                else:
                    print("Usage: simulate --visualize sankey <from>-><to>")
            else:
                print("Visualization types: comparison, sankey <from>-><to>")
        
        else:
            print(f"Unknown simulation action: {action}")
    
    def _run_quick_simulation(self, n_sims: int = 1000, goal_screens: list = None):
        """Run a quick simulation and show key metrics."""
        results = self.simulator.run_simulation(n_sims, use_modified=True, goal_screens=goal_screens)
        
        print(f"\n=== Simulation Results ({n_sims:,} journeys) ===")
        print(f"Average journey length: {results['avg_journey_length']:.1f} screens")
        
        if results['conversion_rates']:
            print("\nConversion rates:")
            for goal, rate in results['conversion_rates'].items():
                print(f"  - {goal}: {rate:.1f}%")
        
        print("\nTop transition paths:")
        for i, (path, count) in enumerate(results['top_paths'][:5], 1):
            percentage = (count / n_sims * 100)
            print(f"  {i}. {path}: {percentage:.1f}% ({count:,} times)")
    
    def _run_comparison(self, n_sims: int, goal_screens: list = None):
        """Run comparison between baseline and modified scenarios."""
        comparison = self.simulator.compare_scenarios(n_sims, goal_screens=goal_screens)
        
        print(f"\n=== Scenario Comparison ({n_sims:,} simulations each) ===")
        
        # Journey length comparison
        journey_change = comparison['journey_length_change']
        print(f"\nAverage Journey Length:")
        print(f"  Baseline: {comparison['baseline']['avg_journey_length']:.1f} screens")
        print(f"  Modified: {comparison['modified']['avg_journey_length']:.1f} screens")
        print(f"  Change: {journey_change['absolute']:+.1f} screens ({journey_change['percent']:+.1f}%)")
        
        # Conversion rate comparison
        if comparison['conversion_changes']:
            print(f"\nConversion Rate Changes:")
            for goal, changes in comparison['conversion_changes'].items():
                print(f"  {goal}:")
                print(f"    Baseline: {changes['baseline']:.1f}%")
                print(f"    Modified: {changes['modified']:.1f}%")
                print(f"    Change: {changes['absolute_change']:+.1f}% ({changes['percent_change']:+.1f}%)")
        
        # Show modifications that led to these results
        modifications = self.simulator.get_modification_summary()
        if modifications:
            print(f"\nModifications applied:")
            for mod in modifications:
                print(f"  {mod['from_screen']} -> {mod['to_screen']}: {mod['baseline_probability']:.3f} -> {mod['modified_probability']:.3f} ({mod['percent_change']:+.1f}%)")
    
    def _show_modifications(self):
        """Show current modifications to transition probabilities."""
        modifications = self.simulator.get_modification_summary()
        
        if not modifications:
            print("\nNo modifications currently applied (using baseline probabilities)")
            return
        
        print(f"\nCurrent modifications ({len(modifications)} changes):")
        for i, mod in enumerate(modifications, 1):
            print(f"  {i}. {mod['from_screen']} -> {mod['to_screen']}:")
            print(f"     Baseline: {mod['baseline_probability']:.3f}")
            print(f"     Modified: {mod['modified_probability']:.3f}")
            print(f"     Change: {mod['percent_change']:+.1f}%")
    
    def run(self):
        print("Interactive Flow Analyzer - Type 'help' for commands, 'exit' to quit")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command == 'exit':
                    print("Goodbye!")
                    break
                elif command == 'help':
                    self.show_help()
                elif command == 'stats':
                    self.handle_stats()
                elif command == 'fields':
                    self.handle_fields()
                elif command == 'simulate':
                    self.handle_simulate(parts)
                elif command == 'most-common-to':
                    if len(parts) < 2:
                        print("Usage: most-common-to <screen> [--where field=value]")
                        continue
                    screen = parts[1]
                    
                    # Parse --where clauses
                    field_filter = self._parse_where_clause(parts)
                    
                    # Extract top_n if provided (but not part of --where)
                    remaining_parts = [p for p in parts[2:] if p != '--where' and '=' not in p]
                    top_n = int(remaining_parts[0]) if remaining_parts and remaining_parts[0].isdigit() else 5
                    
                    self.handle_most_common_to(screen, top_n, field_filter)
                elif command == 'paths':
                    if len(parts) < 3:
                        print("Usage: paths <start> <end>")
                        continue
                    start, end = parts[1], parts[2]
                    top_n = int(parts[3]) if len(parts) > 3 else 5
                    self.handle_paths(start, end, top_n)
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Interactive Flow Path Analysis')
    parser.add_argument('--data', type=str, required=True,
                      help='Data source (file path or URL)')
    parser.add_argument('--pre-aggregated', action='store_true',
                      help='Treat data as pre-aggregated paths with counts instead of raw events')
    parser.add_argument('--no-progress', action='store_true',
                      help='Disable progress bars and verbose loading messages')
    parser.add_argument('--quiet', '-q', action='store_true',
                      help='Minimal output, disable all progress indicators')
    
    args = parser.parse_args()
    
    # Determine progress settings
    show_progress = not args.no_progress and not args.quiet
    
    try:
        analyzer = InteractiveFlowAnalyzer(args.data, is_pre_aggregated=args.pre_aggregated, show_progress=show_progress)
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()