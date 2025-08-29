#!/usr/bin/env python3
"""
Demo script to showcase progress tracking functionality.
"""

from flow_analyzer import FlowPathAnalyzer


def demo_progress_callback(message: str, percentage: float):
    """Custom progress callback that shows percentage and message."""
    bar_length = 30
    filled_length = int(bar_length * percentage / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"\r[{bar}] {percentage:.1f}% - {message}", end='', flush=True)


def demo_with_progress():
    """Demo with custom progress callback."""
    print("=== Demo with Custom Progress Callback ===")
    analyzer = FlowPathAnalyzer(show_progress=False)  # Disable built-in progress
    
    # Load data with custom progress tracking
    analyzer.load_data('example_data_with_context.json', progress_callback=demo_progress_callback)
    print("\n")  # New line after progress bar
    
    # Show some results
    stats = analyzer.get_statistics()
    print(f"Loaded {stats['total_sessions']} sessions with {stats['unique_screens']} unique screens")


def demo_with_tqdm():
    """Demo with built-in tqdm progress bars."""
    print("\n=== Demo with Built-in Progress Bars ===")
    analyzer = FlowPathAnalyzer(show_progress=True)
    analyzer.load_data('example_data_with_context.json')
    
    stats = analyzer.get_statistics()
    print(f"Loaded {stats['total_sessions']} sessions with {stats['unique_screens']} unique screens")


def demo_quiet():
    """Demo with no progress output."""
    print("\n=== Demo with Quiet Mode ===")
    analyzer = FlowPathAnalyzer(show_progress=False)
    analyzer.load_data('example_data_with_context.json')
    
    stats = analyzer.get_statistics()
    print(f"Loaded {stats['total_sessions']} sessions with {stats['unique_screens']} unique screens")


if __name__ == "__main__":
    demo_with_progress()
    demo_with_tqdm()
    demo_quiet()
    print("\nProgress tracking demo complete!")