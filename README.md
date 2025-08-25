# Flow Path Analyzer

A Python application for analyzing and visualizing user flow paths through screen views, showing the most frequent paths between screens.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Analyze paths from Home to OrderConfirmation:
```bash
python main.py --start Home --end OrderConfirmation --data example_data.json
```

### With All Visualizations
```bash
python main.py --start Home --end Checkout --viz-type all --show-stats
```

### Command Line Options

- `--data`: Path to data file (JSON or CSV format)
- `--start`: Starting screen name
- `--end`: Ending screen name  
- `--top-paths`: Number of top paths to display (default: 5)
- `--viz-type`: Type of visualization (flow, sankey, heatmap, network, all)
- `--output-dir`: Directory to save visualizations (default: output)
- `--show-stats`: Display flow statistics

### Demo Mode

Run without arguments to see a demo:
```bash
python main.py
```

## Data Format

### JSON Format
```json
[
    {
        "session_id": "user1_session1",
        "screen": "Home", 
        "timestamp": "2024-01-15T10:00:00"
    }
]
```

### CSV Format
```csv
session_id,screen,timestamp
user1_session1,Home,2024-01-15T10:00:00
user1_session1,Products,2024-01-15T10:00:30
```

## Features

- **Path Analysis**: Find most frequent paths between any two screens
- **Flow Diagram**: Network diagram showing transitions with weights
- **Sankey Diagram**: Interactive flow visualization (HTML)
- **Heatmap**: Transition matrix visualization (HTML)
- **Network Graph**: Interactive network of all screen transitions (HTML)
- **Statistics**: Session counts, screen visits, average path lengths

## API Usage

```python
from flow_analyzer import FlowPathAnalyzer
from visualizer import FlowPathVisualizer

# Initialize analyzer
analyzer = FlowPathAnalyzer()
analyzer.load_data('your_data.json')

# Find paths between screens
paths = analyzer.find_paths('Home', 'Checkout')
for path, count in paths[:5]:
    print(f"{' -> '.join(path)}: {count}")

# Get statistics
stats = analyzer.get_statistics()

# Create visualizations
viz = FlowPathVisualizer(analyzer)
viz.create_flow_diagram('Home', 'Checkout', output_file='flow.png')
viz.create_interactive_sankey('Home', 'Checkout', output_file='sankey.html')
```