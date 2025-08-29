#!/usr/bin/env python3

import argparse
import json
from flow_analyzer import FlowPathAnalyzer
from visualizer import FlowPathVisualizer


def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize user flow paths')
    parser.add_argument('--data', type=str, default='example_data_with_context.json',
                      help='Path to data file (JSON or CSV)')
    parser.add_argument('--start', type=str, 
                      help='Starting screen name')
    parser.add_argument('--end', type=str, 
                      help='Ending screen name')
    parser.add_argument('--most-common-to', type=str,
                      help='Find most common paths leading to this screen (no start screen required)')
    parser.add_argument('--top-paths', type=int, default=5,
                      help='Number of top paths to display (default: 5)')
    parser.add_argument('--viz-type', type=str, default='all',
                      choices=['flow', 'sankey', 'heatmap', 'network', 'all', 'none'],
                      help='Type of visualization to create (use "none" to skip visualization)')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save visualizations')
    parser.add_argument('--show-stats', action='store_true',
                      help='Show flow statistics')
    parser.add_argument('--include-context', action='store_true',
                      help='Include preceding context steps in the analysis')
    parser.add_argument('--context-steps', type=int, default=3,
                      help='Number of preceding steps to include as context (default: 3)')
    parser.add_argument('--split-by-source', action='store_true',
                      help='Split analysis by traffic source (SEO, direct, campaign, etc.)')
    parser.add_argument('--split-by', type=str, 
                      help='Field name to split analysis by (e.g., user_type, campaign_id, device)')
    parser.add_argument('--split-by-multiple', type=str, nargs='+',
                      help='Multiple field names to split by (creates combinations)')
    parser.add_argument('--list-fields', action='store_true',
                      help='List all available fields for segmentation')
    parser.add_argument('--interactive', action='store_true',
                      help='Start interactive mode for multiple queries')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    analyzer = FlowPathAnalyzer()
    analyzer.load_data(args.data)
    
    # Handle interactive mode
    if args.interactive:
        from interactive import InteractiveFlowAnalyzer
        interactive_analyzer = InteractiveFlowAnalyzer(args.data)
        interactive_analyzer.run()
        return
    
    # Handle listing available fields
    if args.list_fields:
        available_fields = analyzer.get_available_fields()
        if available_fields:
            print("\nAvailable fields for segmentation:")
            for field in available_fields:
                values = analyzer.get_field_values(field)
                print(f"  - {field}: {len(values)} unique values")
                if len(values) <= 10:
                    print(f"    Values: {', '.join(map(str, values[:10]))}")
                else:
                    print(f"    Sample values: {', '.join(map(str, values[:5]))}...")
        else:
            print("\nNo segmentation fields found in the data.")
        return
    
    # Handle most common paths to a screen
    if args.most_common_to:
        print(f"\nFinding most common paths leading to '{args.most_common_to}'...")
        most_common_paths = analyzer.find_most_common_paths_to_screen(
            args.most_common_to, 
            top_n=args.top_paths
        )
        
        if not most_common_paths:
            print(f"No paths found leading to '{args.most_common_to}'")
            return
        
        print(f"\nTop {len(most_common_paths)} paths leading to '{args.most_common_to}':")
        for i, (path, count) in enumerate(most_common_paths, 1):
            print(f"{i}. {' -> '.join(path)} (Count: {count})")
        return
    
    # Check if start and end are provided for path analysis
    if not args.start or not args.end:
        parser.error("--start and --end are required for path analysis (or use --most-common-to for end-screen-only analysis)")
    
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
    
    # Handle dynamic field splitting
    if args.split_by:
        print(f"\nAnalyzing paths by {args.split_by}...")
        results_by_field = analyzer.find_paths_by_field(
            args.start, args.end, args.split_by,
            include_context=args.include_context,
            context_steps=args.context_steps
        )
        
        if not results_by_field:
            print(f"No paths found from '{args.start}' to '{args.end}'")
            return
        
        # Display results for each field value
        for value, result in sorted(results_by_field.items(), 
                                   key=lambda x: x[1]['total_sessions'], 
                                   reverse=True):
            print(f"\n{'='*60}")
            print(f"{args.split_by}: {value}")
            print(f"Total Sessions: {result['total_sessions']}")
            print(f"Sessions with Path: {result['sessions_with_path']}")
            print(f"Conversion Rate: {result['conversion_rate']:.2f}%")
            
            if result['paths']:
                if args.include_context and 'full_journeys' in result:
                    print(f"\nTop {min(args.top_paths, len(result['full_journeys']))} complete journeys:")
                    for i, (journey, count) in enumerate(result['full_journeys'][:args.top_paths], 1):
                        start_idx = journey.index(args.start) if args.start in journey else 0
                        formatted_journey = []
                        for j, screen in enumerate(journey):
                            if j < start_idx:
                                formatted_journey.append(f"[{screen}]")
                            else:
                                formatted_journey.append(screen)
                        print(f"  {i}. {' -> '.join(formatted_journey)} (Count: {count})")
                else:
                    print(f"\nTop {min(args.top_paths, len(result['paths']))} paths:")
                    for i, (path, count) in enumerate(result['paths'][:args.top_paths], 1):
                        print(f"  {i}. {' -> '.join(path)} (Count: {count})")
        
        # Store paths for visualization
        paths = []
        for field_result in results_by_field.values():
            paths.extend(field_result['paths'])
        
        # Deduplicate paths
        path_dict = {}
        for path, count in paths:
            path_str = ' -> '.join(path)
            path_dict[path_str] = path_dict.get(path_str, 0) + count
        
        paths = [(path.split(' -> '), count) for path, count in 
                sorted(path_dict.items(), key=lambda x: x[1], reverse=True)]
    
    elif args.split_by_multiple:
        print(f"\nAnalyzing paths by combination of: {', '.join(args.split_by_multiple)}...")
        results_by_combo = analyzer.find_paths_by_multiple_fields(
            args.start, args.end, args.split_by_multiple,
            include_context=args.include_context,
            context_steps=args.context_steps
        )
        
        if not results_by_combo:
            print(f"No paths found from '{args.start}' to '{args.end}'")
            return
        
        # Display top combinations
        sorted_combos = sorted(results_by_combo.items(), 
                             key=lambda x: x[1]['conversion_rate'], 
                             reverse=True)
        
        for combo_key, result in sorted_combos[:10]:  # Show top 10 combinations
            if result['sessions_with_path'] > 0:
                print(f"\n{'='*60}")
                print(f"Combination: {combo_key}")
                print(f"Total Sessions: {result['total_sessions']}")
                print(f"Sessions with Path: {result['sessions_with_path']}")
                print(f"Conversion Rate: {result['conversion_rate']:.2f}%")
                
                if result['paths']:
                    if args.include_context and 'full_journeys' in result:
                        print(f"\nTop complete journeys (with context):")
                        for i, (journey, count) in enumerate(result['full_journeys'][:3], 1):
                            start_idx = journey.index(args.start) if args.start in journey else 0
                            formatted_journey = []
                            for j, screen in enumerate(journey):
                                if j < start_idx:
                                    formatted_journey.append(f"[{screen}]")
                                else:
                                    formatted_journey.append(screen)
                            print(f"  {i}. {' -> '.join(formatted_journey)} (Count: {count})")
                    else:
                        print(f"\nTop paths:")
                        for i, (path, count) in enumerate(result['paths'][:3], 1):
                            print(f"  {i}. {' -> '.join(path)} (Count: {count})")
        
        # Aggregate paths for visualization
        paths = []
        for combo_result in results_by_combo.values():
            paths.extend(combo_result['paths'])
        
        path_dict = {}
        for path, count in paths:
            path_str = ' -> '.join(path)
            path_dict[path_str] = path_dict.get(path_str, 0) + count
        
        paths = [(path.split(' -> '), count) for path, count in 
                sorted(path_dict.items(), key=lambda x: x[1], reverse=True)]
    
    elif args.split_by_source:
        print("\nAnalyzing paths by traffic source...")
        results_by_source = analyzer.find_paths_by_traffic_source(
            args.start, args.end, 
            include_context=args.include_context,
            context_steps=args.context_steps
        )
        
        if not results_by_source:
            print(f"No paths found from '{args.start}' to '{args.end}'")
            return
        
        # Display results for each traffic source
        for source, result in sorted(results_by_source.items(), 
                                    key=lambda x: x[1]['total_sessions'], 
                                    reverse=True):
            print(f"\n{'='*60}")
            print(f"Traffic Source: {source.upper()}")
            print(f"Total Sessions: {result['total_sessions']}")
            print(f"Sessions with Path: {result['sessions_with_path']}")
            print(f"Conversion Rate: {result['conversion_rate']:.2f}%")
            
            if result['paths']:
                if args.include_context and 'full_journeys' in result:
                    print(f"\nTop {min(args.top_paths, len(result['full_journeys']))} complete journeys:")
                    for i, (journey, count) in enumerate(result['full_journeys'][:args.top_paths], 1):
                        start_idx = journey.index(args.start) if args.start in journey else 0
                        formatted_journey = []
                        for j, screen in enumerate(journey):
                            if j < start_idx:
                                formatted_journey.append(f"[{screen}]")
                            else:
                                formatted_journey.append(screen)
                        print(f"  {i}. {' -> '.join(formatted_journey)} (Count: {count})")
                else:
                    print(f"\nTop {min(args.top_paths, len(result['paths']))} paths:")
                    for i, (path, count) in enumerate(result['paths'][:args.top_paths], 1):
                        print(f"  {i}. {' -> '.join(path)} (Count: {count})")
        
        # Store the paths for visualization
        # We'll use the combined paths from all sources for the default visualizations
        paths = []
        for source_result in results_by_source.values():
            paths.extend(source_result['paths'])
        
        # Sort and deduplicate
        path_dict = {}
        for path, count in paths:
            path_str = ' -> '.join(path)
            path_dict[path_str] = path_dict.get(path_str, 0) + count
        
        paths = [(path.split(' -> '), count) for path, count in 
                sorted(path_dict.items(), key=lambda x: x[1], reverse=True)]
        
    elif args.include_context:
        print(f"Including up to {args.context_steps} preceding context steps...")
        result = analyzer.find_paths_with_context(args.start, args.end, 
                                                 context_steps=args.context_steps)
        paths = result['paths']
        full_journeys = result['full_journeys']
        
        if not paths:
            print(f"No paths found from '{args.start}' to '{args.end}'")
            return
        
        print(f"\nTop {min(args.top_paths, len(full_journeys))} complete journeys (with context):")
        for i, (journey, count) in enumerate(full_journeys[:args.top_paths], 1):
            # Find where the main path starts
            start_idx = journey.index(args.start) if args.start in journey else 0
            
            # Format the journey with visual distinction
            formatted_journey = []
            for j, screen in enumerate(journey):
                if j < start_idx:
                    formatted_journey.append(f"[{screen}]")  # Context in brackets
                else:
                    formatted_journey.append(screen)  # Main path normal
            
            print(f"{i}. {' -> '.join(formatted_journey)} (Count: {count})")
        
        print(f"\nMain paths (without context):")
        for i, (path, count) in enumerate(paths[:args.top_paths], 1):
            print(f"{i}. {' -> '.join(path)} (Count: {count})")
    else:
        paths = analyzer.find_paths(args.start, args.end)
        
        if not paths:
            print(f"No paths found from '{args.start}' to '{args.end}'")
            return
        
        print(f"\nTop {min(args.top_paths, len(paths))} paths:")
        for i, (path, count) in enumerate(paths[:args.top_paths], 1):
            print(f"{i}. {' -> '.join(path)} (Count: {count})")
    
    # Skip visualization if requested
    if args.viz_type == 'none':
        return
        
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
        if args.split_by_multiple:
            print(f"\nCreating Sankey diagrams by combination of: {', '.join(args.split_by_multiple)}...")
            visualizer.create_sankey_charts_by_multiple_fields(
                args.start, args.end, args.split_by_multiple, args.top_paths,
                args.output_dir
            )
        elif args.split_by:
            print(f"\nCreating Sankey diagrams by {args.split_by}...")
            visualizer.create_sankey_charts_by_field(
                args.start, args.end, args.split_by, args.top_paths,
                args.output_dir,
                include_context=args.include_context,
                context_steps=args.context_steps
            )
        elif args.split_by_source:
            print("\nCreating Sankey diagrams by traffic source...")
            visualizer.create_sankey_charts_by_field(
                args.start, args.end, 'traffic_source', args.top_paths,
                args.output_dir
            )
        elif args.include_context:
            print("Creating Sankey diagram...")
            visualizer.create_interactive_sankey_with_context(
                args.start, args.end, args.top_paths,
                context_steps=args.context_steps,
                output_file=f"{args.output_dir}/sankey_diagram.html"
            )
        else:
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