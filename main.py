#!/usr/bin/env python3

import argparse
import json
from flow_analyzer import FlowPathAnalyzer
from visualizer import FlowPathVisualizer


def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize user flow paths')
    parser.add_argument('--data', type=str, default='example_data.json',
                      help='Path to data file (JSON or CSV)')
    parser.add_argument('--start', type=str, required=True,
                      help='Starting screen name')
    parser.add_argument('--end', type=str, required=True,
                      help='Ending screen name')
    parser.add_argument('--top-paths', type=int, default=5,
                      help='Number of top paths to display (default: 5)')
    parser.add_argument('--viz-type', type=str, default='all',
                      choices=['flow', 'sankey', 'heatmap', 'network', 'all'],
                      help='Type of visualization to create')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save visualizations')
    parser.add_argument('--show-stats', action='store_true',
                      help='Show flow statistics')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    analyzer = FlowPathAnalyzer()
    analyzer.load_data(args.data)
    
    if args.show_stats:
        print("\n=== Flow Statistics ===")
        stats = analyzer.get_statistics()
        print(f"Total Sessions: {stats['total_sessions']}")
        print(f"Unique Screens: {stats['unique_screens']}")
        print(f"Average Path Length: {stats['average_path_length']}")
        print(f"Total Transitions: {stats['total_transitions']}")
        print("\nMost Visited Screens:")
        for screen, count in stats['most_visited_screens'][:5]:
            print(f"  - {screen}: {count} visits")
    
    print(f"\nFinding paths from '{args.start}' to '{args.end}'...")
    paths = analyzer.find_paths(args.start, args.end)
    
    if not paths:
        print(f"No paths found from '{args.start}' to '{args.end}'")
        return
    
    print(f"\nTop {min(args.top_paths, len(paths))} paths:")
    for i, (path, count) in enumerate(paths[:args.top_paths], 1):
        print(f"{i}. {' -> '.join(path)} (Count: {count})")
    
    visualizer = FlowPathVisualizer(analyzer)
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.viz_type in ['flow', 'all']:
        print("\nCreating flow diagram...")
        visualizer.create_flow_diagram(
            args.start, args.end, args.top_paths,
            f"{args.output_dir}/flow_diagram.png"
        )
    
    if args.viz_type in ['sankey', 'all']:
        print("Creating Sankey diagram...")
        visualizer.create_interactive_sankey(
            args.start, args.end, args.top_paths,
            f"{args.output_dir}/sankey_diagram.html"
        )
    
    if args.viz_type in ['heatmap', 'all']:
        print("Creating transition heatmap...")
        visualizer.create_heatmap(f"{args.output_dir}/heatmap.html")
    
    if args.viz_type in ['network', 'all']:
        print("Creating network graph...")
        visualizer.create_network_graph(
            min_transitions=1,
            output_file=f"{args.output_dir}/network_graph.html"
        )
    
    print("\nVisualization complete! Check the output directory for results.")


def demo():
    print("Running demo with example data...")
    
    analyzer = FlowPathAnalyzer()
    
    sample_data = [
        {"session_id": "s1", "screen": "Home", "timestamp": "2024-01-01T10:00:00"},
        {"session_id": "s1", "screen": "Products", "timestamp": "2024-01-01T10:01:00"},
        {"session_id": "s1", "screen": "Cart", "timestamp": "2024-01-01T10:02:00"},
        {"session_id": "s1", "screen": "Checkout", "timestamp": "2024-01-01T10:03:00"},
        
        {"session_id": "s2", "screen": "Home", "timestamp": "2024-01-01T11:00:00"},
        {"session_id": "s2", "screen": "Search", "timestamp": "2024-01-01T11:01:00"},
        {"session_id": "s2", "screen": "Products", "timestamp": "2024-01-01T11:02:00"},
        {"session_id": "s2", "screen": "Cart", "timestamp": "2024-01-01T11:03:00"},
        {"session_id": "s2", "screen": "Checkout", "timestamp": "2024-01-01T11:04:00"},
        
        {"session_id": "s3", "screen": "Home", "timestamp": "2024-01-01T12:00:00"},
        {"session_id": "s3", "screen": "Products", "timestamp": "2024-01-01T12:01:00"},
        {"session_id": "s3", "screen": "ProductDetail", "timestamp": "2024-01-01T12:02:00"},
        {"session_id": "s3", "screen": "Cart", "timestamp": "2024-01-01T12:03:00"},
        {"session_id": "s3", "screen": "Checkout", "timestamp": "2024-01-01T12:04:00"},
    ]
    
    analyzer.load_data(sample_data)
    
    print("\n=== Statistics ===")
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        if key != 'most_visited_screens':
            print(f"{key}: {value}")
    
    print("\n=== Most Common Paths ===")
    common_paths = analyzer.get_most_common_paths(top_n=3)
    for path, count in common_paths:
        print(f"{' -> '.join(path)}: {count} occurrences")
    
    print("\n=== Paths from Home to Checkout ===")
    paths = analyzer.find_paths("Home", "Checkout")
    for path, count in paths[:3]:
        print(f"{' -> '.join(path)}: {count} occurrences")
    
    visualizer = FlowPathVisualizer(analyzer)
    print("\nCreating visualizations...")
    visualizer.create_flow_diagram("Home", "Checkout", top_n_paths=3)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("No arguments provided. Running demo...")
        demo()
    elif "--demo" in sys.argv:
        demo()
    else:
        main()