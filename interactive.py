#!/usr/bin/env python3

import argparse
from flow_analyzer import FlowPathAnalyzer


class InteractiveFlowAnalyzer:
    def __init__(self, data_source: str, is_pre_aggregated: bool = False):
        self.analyzer = FlowPathAnalyzer(is_pre_aggregated=is_pre_aggregated)
        print(f"Loading data from {data_source}...")
        self.analyzer.load_data(data_source)
        print("Data loaded successfully!")
        
    def show_help(self):
        print("""
Available commands:
  most-common-to <screen> [--where field=value]  - Find most common paths to a screen with optional filtering
  paths <start> <end>                           - Find paths from start to end screen
  stats                                         - Show flow statistics
  fields                                        - List available segmentation fields
  help                                          - Show this help message
  exit                                          - Exit the interactive session

Examples:
  most-common-to Cart
  most-common-to Checkout --where purchased=true
  most-common-to Products --where device=mobile
  paths Home Checkout
  stats
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
    
    args = parser.parse_args()
    
    try:
        analyzer = InteractiveFlowAnalyzer(args.data, is_pre_aggregated=args.pre_aggregated)
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()