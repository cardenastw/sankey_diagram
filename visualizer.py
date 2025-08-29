import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import pydot
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class FlowPathVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.show_progress = getattr(analyzer, 'show_progress', True)
        
    def create_flow_diagram(self, start_screen: str, end_screen: str, 
                           top_n_paths: int = 5, output_file: Optional[str] = None):
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Using interactive Plotly network graph instead.")
            return self.create_network_graph_for_paths(start_screen, end_screen, top_n_paths, output_file)
            
        paths = self.analyzer.find_paths(start_screen, end_screen)[:top_n_paths]
        
        if not paths:
            print(f"No paths found from '{start_screen}' to '{end_screen}'")
            return
        
        G = nx.DiGraph()
        
        for path, count in paths:
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                
                if G.has_edge(from_node, to_node):
                    G[from_node][to_node]['weight'] += count
                else:
                    G.add_edge(from_node, to_node, weight=count)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=3000, alpha=0.9)
        
        nx.draw_networkx_nodes(G, pos, nodelist=[start_screen], 
                             node_color='green', node_size=3000, alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=[end_screen], 
                             node_color='red', node_size=3000, alpha=0.9)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(G, pos, edgelist=edges,
                              width=[5 * w / max_weight for w in weights],
                              alpha=0.6, edge_color='gray',
                              connectionstyle='arc3,rad=0.1',
                              arrowsize=20, arrowstyle='->')
        
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title(f"Flow Paths from '{start_screen}' to '{end_screen}'")
        plt.axis('off')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Flow diagram saved to {output_file}")
        else:
            plt.show()
    
    def create_network_graph_for_paths(self, start_screen: str, end_screen: str,
                                      top_n_paths: int = 5, output_file: Optional[str] = None):
        paths = self.analyzer.find_paths(start_screen, end_screen)[:top_n_paths]
        
        if not paths:
            print(f"No paths found from '{start_screen}' to '{end_screen}'")
            return
        
        G = nx.DiGraph()
        
        for path, count in paths:
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                
                if G.has_edge(from_node, to_node):
                    G[from_node][to_node]['weight'] += count
                else:
                    G.add_edge(from_node, to_node, weight=count)
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=min(weight/2, 10), color='gray'),
                hoverinfo='text',
                hovertext=f"{edge[0]} → {edge[1]}: {weight} paths"
            ))
        
        node_colors = []
        for node in G.nodes():
            if node == start_screen:
                node_colors.append('green')
            elif node == end_screen:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=[node for node in G.nodes()],
            mode='markers+text',
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=30,
                color=node_colors,
                line_width=2
            )
        )
        
        fig = go.Figure(data=edge_trace + [node_trace],
                       layout=go.Layout(
                           title=f"Flow Paths from '{start_screen}' to '{end_screen}'",
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=700,
                           width=1000
                       ))
        
        if output_file:
            if output_file.endswith('.png'):
                output_file = output_file.replace('.png', '.html')
            fig.write_html(output_file)
            print(f"Interactive flow diagram saved to {output_file}")
        else:
            fig.show()
    
    def create_interactive_sankey(self, start_screen: str, end_screen: str, 
                                 top_n_paths: int = 10, output_file: Optional[str] = None, 
                                 simplified: bool = True):
        if self.show_progress:
            print(f"Creating Sankey diagram for {start_screen} → {end_screen}...")
        
        paths = self.analyzer.find_paths(start_screen, end_screen)[:top_n_paths]
        
        if not paths:
            print(f"No paths found from '{start_screen}' to '{end_screen}'")
            return
        
        if simplified:
            return self._create_simplified_sankey(paths, start_screen, end_screen, output_file)
        
        # Find the maximum path length to determine layout
        max_path_length = max(len(path) for path, _ in paths)
        
        # Create position-based node mapping
        nodes = []
        node_positions = {}
        node_dict = {}
        links = []
        
        # First pass: collect all unique nodes at each position
        position_nodes = defaultdict(set)
        for path, count in paths:
            for i, screen in enumerate(path):
                position_nodes[i].add(screen)
        
        # Second pass: assign indices to nodes based on position
        for position in sorted(position_nodes.keys()):
            for screen in sorted(position_nodes[position]):
                node_key = f"{screen}_{position}"
                if node_key not in node_dict:
                    node_dict[node_key] = len(nodes)
                    node_positions[len(nodes)] = position / (max_path_length - 1) if max_path_length > 1 else 0.5
                    # Add position indicator to label for clarity
                    if position == 0:
                        label = f"{screen} (Start)"
                    elif screen == end_screen:
                        label = f"{screen} (End)"
                    else:
                        label = f"{screen} (Step {position})"
                    nodes.append(label)
        
        # Third pass: create links
        link_dict = defaultdict(int)
        for path, count in paths:
            for i in range(len(path) - 1):
                source_key = f"{path[i]}_{i}"
                target_key = f"{path[i+1]}_{i+1}"
                link_key = (node_dict[source_key], node_dict[target_key])
                link_dict[link_key] += count
        
        # Convert to links list
        for (source, target), value in link_dict.items():
            links.append({
                'source': source,
                'target': target,
                'value': value
            })
        
        # Assign colors to nodes based on position
        node_colors = []
        for idx in range(len(nodes)):
            pos = node_positions[idx]
            if pos == 0:
                node_colors.append('rgba(34, 139, 34, 0.8)')  # Green for start
            elif pos == 1:
                node_colors.append('rgba(220, 20, 60, 0.8)')  # Red for end
            else:
                # Gradient from blue to purple for intermediate steps
                blue_component = int(255 * (1 - pos))
                node_colors.append(f'rgba(100, 100, {blue_component}, 0.8)')
        
        # Calculate node x positions explicitly
        x_positions = [node_positions[i] for i in range(len(nodes))]
        
        # Calculate y positions to spread nodes vertically at each step
        y_positions = []
        for position in sorted(set(node_positions.values())):
            nodes_at_position = [i for i, pos in node_positions.items() if pos == position]
            num_nodes = len(nodes_at_position)
            if num_nodes == 1:
                y_positions.extend([0.5] * num_nodes)
            else:
                step = 0.8 / (num_nodes - 1) if num_nodes > 1 else 0
                y_values = [0.1 + i * step for i in range(num_nodes)]
                y_positions.extend(y_values)
        
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=20,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=node_colors,
                x=x_positions,
                y=y_positions[:len(nodes)]
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                color="rgba(100, 100, 100, 0.3)",
                label=[f"Count: {link['value']}" for link in links]
            )
        )])
        
        fig.update_layout(
            title=f"Flow Paths from '{start_screen}' to '{end_screen}' (Sankey Diagram)",
            font_size=11,
            height=max(600, len(nodes) * 30),
            width=max(1200, max_path_length * 150),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Interactive Sankey diagram saved to {output_file}")
        else:
            fig.show()
    
    def create_interactive_sankey_with_context(self, start_screen: str, end_screen: str,
                                              top_n_paths: int = 10, context_steps: int = 3,
                                              output_file: Optional[str] = None):
        """Create a Sankey diagram including preceding context steps."""
        
        if self.show_progress:
            print(f"Creating Sankey diagram with context for {start_screen} → {end_screen}...")
        
        # Get paths with context
        result = self.analyzer.find_paths_with_context(start_screen, end_screen, 
                                                      max_length=10, 
                                                      context_steps=context_steps)
        
        full_journeys = result['full_journeys'][:top_n_paths]
        
        if not full_journeys:
            print(f"No paths found from '{start_screen}' to '{end_screen}'")
            return
        
        # Find the maximum journey length and identify where the main path starts
        max_journey_length = max(len(journey) for journey, _ in full_journeys)
        
        # Identify the position where start_screen typically appears
        start_positions = []
        for journey, _ in full_journeys:
            if start_screen in journey:
                start_positions.append(journey.index(start_screen))
        
        # Build node mapping with position awareness
        node_levels = {}
        for journey, _ in full_journeys:
            start_idx = journey.index(start_screen) if start_screen in journey else 0
            for i, node in enumerate(journey):
                # Assign negative levels for context, positive for main path
                if i < start_idx:
                    level = i - start_idx  # Negative for context
                else:
                    level = i - start_idx  # 0 or positive for main path
                
                if node not in node_levels:
                    node_levels[node] = level
                else:
                    # Keep the most common level for each node
                    node_levels[node] = min(node_levels[node], level)
        
        # Normalize levels to 0-1 range
        min_level = min(node_levels.values())
        max_level = max(node_levels.values())
        level_range = max_level - min_level if max_level != min_level else 1
        
        # Create ordered node list
        nodes_by_level = defaultdict(list)
        for node, level in node_levels.items():
            nodes_by_level[level].append(node)
        
        node_list = []
        node_to_idx = {}
        x_positions = []
        
        for level in sorted(nodes_by_level.keys()):
            for node in sorted(nodes_by_level[level]):
                node_to_idx[node] = len(node_list)
                node_list.append(node)
                # Calculate x position (0 to 1)
                x_pos = (level - min_level) / level_range
                x_positions.append(x_pos)
        
        # Build flow aggregation
        flow_pairs = defaultdict(int)
        for journey, count in full_journeys:
            for i in range(len(journey) - 1):
                flow_pairs[(journey[i], journey[i+1])] += count
        
        # Create links
        links = []
        for (source, target), value in flow_pairs.items():
            if source in node_to_idx and target in node_to_idx:
                links.append({
                    'source': node_to_idx[source],
                    'target': node_to_idx[target],
                    'value': value
                })
        
        # Color nodes based on their role
        node_colors = []
        node_labels = []
        for node in node_list:
            level = node_levels[node]
            
            if node == start_screen:
                node_colors.append('rgba(46, 125, 50, 0.9)')  # Green for start
                node_labels.append(f"{node} (START)")
            elif node == end_screen:
                node_colors.append('rgba(198, 40, 40, 0.9)')  # Red for end
                node_labels.append(f"{node} (END)")
            elif level < 0:
                # Context nodes (before start) - light gray/blue
                node_colors.append('rgba(150, 150, 200, 0.7)')
                node_labels.append(f"{node} (context)")
            else:
                # Main path nodes - blue gradient
                intensity = 100 + int(155 * (level / max_level)) if max_level > 0 else 180
                node_colors.append(f'rgba(66, 100, {intensity}, 0.8)')
                node_labels.append(node)
        
        # Create link colors - lighter for context transitions
        link_colors = []
        for link in links:
            source_node = node_list[link['source']]
            target_node = node_list[link['target']]
            source_level = node_levels[source_node]
            target_level = node_levels[target_node]
            
            if source_level < 0 or (target_level == 0 and target_node != start_screen):
                # Context transition
                link_colors.append('rgba(180, 180, 200, 0.3)')
            else:
                # Main path transition
                link_colors.append('rgba(100, 150, 200, 0.4)')
        
        # Create the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=30,
                thickness=20,
                line=dict(color="white", width=2),
                label=node_labels,
                color=node_colors,
                x=x_positions,
                hovertemplate='<b>%{label}</b><br>Position: %{customdata}<extra></extra>',
                customdata=[node_levels[node] for node in node_list]
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                color=link_colors,
                hovertemplate='<b>%{source.label} → %{target.label}</b><br>Flow count: %{value}<extra></extra>'
            ),
            textfont=dict(size=11, color='black')
        )])
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"User Journey with Context: {start_screen} → {end_screen}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': 'black'}
            },
            annotations=[
                dict(
                    x=0.01,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='← Context Steps',
                    showarrow=False,
                    font=dict(size=12, color='gray')
                ),
                dict(
                    x=0.99,
                    y=1.05,
                    xref='paper',
                    yref='paper',
                    text='Main Path →',
                    showarrow=False,
                    font=dict(size=12, color='blue')
                )
            ],
            font=dict(size=11),
            height=max(700, len(node_list) * 40),
            width=1600,
            margin=dict(l=10, r=10, t=100, b=40),
            paper_bgcolor='rgba(250, 250, 250, 1)',
            plot_bgcolor='rgba(255, 255, 255, 0)'
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Context-aware Sankey diagram saved to {output_file}")
        else:
            fig.show()
    
    def _create_simplified_sankey(self, paths, start_screen, end_screen, output_file):
        """Create a simplified Sankey diagram with proper flow layout"""
        
        # Calculate node positions based on distance from start
        node_levels = {}
        max_level = 0
        
        # BFS to find the level of each node
        for path, _ in paths:
            for i, node in enumerate(path):
                if node not in node_levels:
                    node_levels[node] = i
                else:
                    node_levels[node] = min(node_levels[node], i)
                max_level = max(max_level, i)
        
        # Create ordered node list based on levels
        nodes_by_level = defaultdict(list)
        for node, level in node_levels.items():
            nodes_by_level[level].append(node)
        
        # Build node list maintaining level order
        node_list = []
        node_to_idx = {}
        
        for level in sorted(nodes_by_level.keys()):
            for node in sorted(nodes_by_level[level]):
                node_to_idx[node] = len(node_list)
                node_list.append(node)
        
        # Calculate x positions based on levels (0 to 1)
        x_positions = []
        for node in node_list:
            level = node_levels[node]
            x_pos = level / max_level if max_level > 0 else 0.5
            x_positions.append(x_pos)
        
        # Build flow aggregation
        flow_pairs = defaultdict(int)
        for path, count in paths:
            for i in range(len(path) - 1):
                flow_pairs[(path[i], path[i+1])] += count
        
        # Create links
        links = []
        for (source, target), value in flow_pairs.items():
            if source in node_to_idx and target in node_to_idx:
                links.append({
                    'source': node_to_idx[source],
                    'target': node_to_idx[target],
                    'value': value
                })
        
        # Color nodes based on role
        node_colors = []
        for node in node_list:
            if node == start_screen:
                node_colors.append('rgba(46, 125, 50, 0.9)')  # Green
            elif node == end_screen:
                node_colors.append('rgba(198, 40, 40, 0.9)')  # Red
            else:
                # Gradient based on position
                level = node_levels[node]
                intensity = 150 + int(105 * (level / max_level))
                node_colors.append(f'rgba(66, 100, {intensity}, 0.8)')
        
        # Create the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=40,
                thickness=25,
                line=dict(color="white", width=2),
                label=node_list,
                color=node_colors,
                x=x_positions,
                hovertemplate='<b>%{label}</b><br>Step %{customdata}<extra></extra>',
                customdata=[node_levels[node] for node in node_list]
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                color='rgba(150, 150, 150, 0.3)',
                hovertemplate='<b>%{source.label} → %{target.label}</b><br>Flow count: %{value}<extra></extra>'
            ),
            textfont=dict(size=12, color='black')
        )])
        
        fig.update_layout(
            title={
                'text': f"User Flow Path: {start_screen} → {end_screen}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': 'black'}
            },
            font=dict(size=11),
            height=max(600, len(node_list) * 50),
            width=1500,
            margin=dict(l=10, r=10, t=80, b=40),
            paper_bgcolor='rgba(250, 250, 250, 1)',
            plot_bgcolor='rgba(255, 255, 255, 0)'
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Simplified Sankey diagram saved to {output_file}")
        else:
            fig.show()
    
    def create_heatmap(self, output_file: Optional[str] = None):
        probability_matrix = self.analyzer.export_transition_probability_matrix()
        
        if probability_matrix.empty:
            print("No transition data available")
            return
        
        # Create text labels showing probabilities as percentages
        text_matrix = probability_matrix.applymap(lambda x: f"{x:.1%}" if x > 0 else "")
        
        # Create hover text with more detail
        hover_text = []
        for i in range(len(probability_matrix.index)):
            hover_row = []
            for j in range(len(probability_matrix.columns)):
                from_screen = probability_matrix.index[i]
                to_screen = probability_matrix.columns[j]
                prob = probability_matrix.iloc[i, j]
                if prob > 0:
                    hover_row.append(f"From: {from_screen}<br>To: {to_screen}<br>Probability: {prob:.2%}")
                else:
                    hover_row.append(f"From: {from_screen}<br>To: {to_screen}<br>No transitions")
            hover_text.append(hover_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=probability_matrix.values,
            x=probability_matrix.columns,
            y=probability_matrix.index,
            colorscale='Blues',
            text=text_matrix.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            zmin=0,
            zmax=1
        ))
        
        fig.update_layout(
            title="Screen Transition Probability Heatmap<br><sub>Shows probability of moving from one screen to another</sub>",
            xaxis_title="To Screen",
            yaxis_title="From Screen",
            height=max(400, len(probability_matrix) * 30),
            width=max(600, len(probability_matrix.columns) * 30),
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'}
        )
        
        # Update colorbar to show percentages
        fig.update_coloraxes(
            colorbar=dict(
                title="Transition<br>Probability",
                tickformat=".0%"
            )
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Probability heatmap saved to {output_file}")
        else:
            fig.show()
    
    def create_network_graph(self, min_transitions: int = 1, 
                           output_file: Optional[str] = None):
        G = nx.DiGraph()
        
        for from_screen, transitions in self.analyzer.screen_transitions.items():
            for to_screen, count in transitions.items():
                if count >= min_transitions:
                    G.add_edge(from_screen, to_screen, weight=count)
        
        if len(G.nodes()) == 0:
            print("No transitions found with the specified minimum threshold")
            return
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=min(weight/5, 10), color='gray'),
                hoverinfo='none'
            ))
        
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=15,
                color=[],
                colorbar=dict(
                    thickness=15,
                    title=dict(text='Connections'),
                    xanchor='left'
                ),
                line_width=2
            )
        )
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])
            
            node_info = f"{node}<br>In: {G.in_degree(node)}<br>Out: {G.out_degree(node)}"
            node_trace['marker']['color'] += tuple([G.degree(node)])
        
        fig = go.Figure(data=edge_trace + [node_trace],
                       layout=go.Layout(
                           title='Screen Flow Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=700,
                           width=1000
                       ))
        
        if output_file:
            fig.write_html(output_file)
            print(f"Network graph saved to {output_file}")
        else:
            fig.show()
    
    def create_sankey_charts_by_field(self, start_screen: str, end_screen: str,
                                     segment_field: str, top_n_paths: int = 10, 
                                     output_dir: str = 'output',
                                     include_context: bool = False, context_steps: int = 3):
        """Create separate Sankey charts for each value of the specified field plus a combined view."""
        
        # Get paths segmented by the specified field
        results_by_value = self.analyzer.find_paths_by_field(
            start_screen, end_screen, segment_field, max_length=10,
            include_context=include_context, context_steps=context_steps
        )
        
        if not results_by_value:
            print(f"No paths found from '{start_screen}' to '{end_screen}'")
            return
        
        # Generate a color palette dynamically based on the number of unique values
        value_list = list(results_by_value.keys())
        colors = self._generate_color_palette(len(value_list))
        value_colors = {value: colors[i] for i, value in enumerate(value_list)}
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # First, create a combined chart with all values
        if self.show_progress:
            print(f"Creating combined Sankey diagram for all {segment_field} values...")
        all_paths = []
        for value, result in results_by_value.items():
            if include_context and 'full_journeys' in result:
                for journey, count in result['full_journeys']:
                    all_paths.append((journey, count, value))
            else:
                for path, count in result['paths']:
                    all_paths.append((path, count, value))
        
        if all_paths:
            self._create_combined_sankey_field(all_paths, start_screen, end_screen, 
                                              segment_field, value_colors, 
                                              f"{output_dir}/sankey_all_{segment_field}.html")
        
        # Then create individual charts for each field value
        if self.show_progress:
            value_items = list(results_by_value.items())
            value_iterator = tqdm(value_items, desc=f"Creating {segment_field} charts", disable=not self.show_progress) if self.show_progress else value_items
        else:
            value_iterator = results_by_value.items()
            
        for value, result in value_iterator:
            if not result['paths']:
                continue
            
            if self.show_progress:
                print(f"Creating Sankey diagram for {segment_field}={value}...")
            
            value_color = value_colors.get(value, 'rgba(128, 128, 128, 0.8)')
            
            # Build node and link structure for this value
            nodes_dict = {}
            links_list = []
            
            # Get top paths or journeys for this value
            if include_context and 'full_journeys' in result:
                top_paths = result['full_journeys'][:top_n_paths]
            else:
                top_paths = result['paths'][:top_n_paths]
            
            for path, count in top_paths:
                # Find where the main path starts (at start_screen)
                start_idx = path.index(start_screen) if start_screen in path else 0
                
                # Create nodes for each screen in the path
                for i, screen in enumerate(path):
                    node_key = f"{screen}_{i}"
                    
                    if node_key not in nodes_dict:
                        # Determine node color and label
                        if i < start_idx:
                            # Context node - light gray/blue
                            color = 'rgba(150, 150, 200, 0.7)'
                            label = f"{screen} (context)"
                        elif screen == start_screen:
                            color = 'rgba(46, 125, 50, 0.9)'  # Green for start
                            label = f"{screen} (START)"
                        elif screen == end_screen:
                            color = 'rgba(198, 40, 40, 0.9)'  # Red for end
                            label = f"{screen} (END)"
                        else:
                            # Use value color with varying intensity
                            intensity_factor = 0.6 + (0.4 * (i - start_idx) / max(len(path) - start_idx - 1, 1))
                            color = value_color.replace('0.8', str(intensity_factor))
                            label = screen
                        
                        nodes_dict[node_key] = {
                            'label': label,
                            'color': color,
                            'position': i
                        }
                
                # Create links between consecutive nodes
                for i in range(len(path) - 1):
                    source_key = f"{path[i]}_{i}"
                    target_key = f"{path[i+1]}_{i+1}"
                    
                    # Check if link exists and update count
                    link_found = False
                    for link in links_list:
                        if link['source_key'] == source_key and link['target_key'] == target_key:
                            link['value'] += count
                            link_found = True
                            break
                    
                    if not link_found:
                        links_list.append({
                            'source_key': source_key,
                            'target_key': target_key,
                            'value': count
                        })
            
            # Sort nodes by position
            sorted_nodes = sorted(nodes_dict.items(), key=lambda x: (x[1]['position'], x[0]))
            
            # Create node list and mapping
            node_list = []
            node_to_idx = {}
            for node_key, node_info in sorted_nodes:
                node_to_idx[node_key] = len(node_list)
                node_list.append(node_info)
            
            # Calculate x positions
            max_position = max(node['position'] for node in node_list) if node_list else 0
            x_positions = []
            for node in node_list:
                x_pos = node['position'] / max_position if max_position > 0 else 0.5
                x_positions.append(x_pos)
            
            # Convert links to use indices
            final_links = []
            for link in links_list:
                if link['source_key'] in node_to_idx and link['target_key'] in node_to_idx:
                    final_links.append({
                        'source': node_to_idx[link['source_key']],
                        'target': node_to_idx[link['target_key']],
                        'value': link['value']
                    })
            
            # Create the Sankey diagram for this source
            fig = go.Figure(data=[go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="white", width=2),
                    label=[node['label'] for node in node_list],
                    color=[node['color'] for node in node_list],
                    x=x_positions
                ),
                link=dict(
                    source=[link['source'] for link in final_links],
                    target=[link['target'] for link in final_links],
                    value=[link['value'] for link in final_links],
                    color=value_color.replace('0.8', '0.3'),
                    hovertemplate='<b>%{source.label} → %{target.label}</b><br>Sessions: %{value}<extra></extra>'
                ),
                textfont=dict(size=12, color='black')
            )])
            
            # Update layout with statistics
            conversion_rate = result['conversion_rate']
            total_sessions = result['total_sessions']
            sessions_with_path = result['sessions_with_path']
            
            fig.update_layout(
                title={
                    'text': f"User Flow: {start_screen} → {end_screen}<br>" +
                           f"<span style='font-size:14px'>{segment_field}: {str(value).replace('_', ' ').title()} | " +
                           f"Conversion Rate: {conversion_rate:.1f}% ({sessions_with_path}/{total_sessions} sessions)</span>",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': 'black'}
                },
                font=dict(size=12),
                height=500,
                width=1400,
                margin=dict(l=10, r=10, t=100, b=40),
                paper_bgcolor='rgba(250, 250, 250, 1)',
                plot_bgcolor='rgba(255, 255, 255, 0)'
            )
            
            # Save the chart
            safe_value_name = str(value).replace(' ', '_').replace('/', '_').replace('=', '_')
            output_file = f"{output_dir}/sankey_{segment_field}_{safe_value_name}.html"
            fig.write_html(output_file)
            print(f"  Saved to {output_file}")
        
        print(f"\nCreated {len([r for r in results_by_value.items() if r[1]['paths']]) + 1} Sankey diagrams in {output_dir}/")
    
    def create_sankey_charts_by_multiple_fields(self, start_screen: str, end_screen: str,
                                               segment_fields: List[str], top_n_paths: int = 10,
                                               output_dir: str = 'output'):
        """Create separate Sankey charts for each combination of multiple field values plus a combined view."""
        
        # Get paths segmented by the multiple fields
        results_by_combo = self.analyzer.find_paths_by_multiple_fields(
            start_screen, end_screen, segment_fields, max_length=10
        )
        
        if not results_by_combo:
            print(f"No paths found from '{start_screen}' to '{end_screen}'")
            return
        
        # Generate a color palette for combinations
        combo_list = list(results_by_combo.keys())
        colors = self._generate_color_palette(min(len(combo_list), 20))  # Limit to 20 colors
        combo_colors = {combo: colors[i % len(colors)] for i, combo in enumerate(combo_list)}
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # First, create a combined chart with all combinations
        print(f"Creating combined Sankey diagram for all combinations...")
        all_paths = []
        for combo_key, result in results_by_combo.items():
            for path, count in result['paths']:
                all_paths.append((path, count, combo_key))
        
        if all_paths:
            self._create_combined_sankey_multiple_fields(all_paths, start_screen, end_screen,
                                                        segment_fields, combo_colors,
                                                        f"{output_dir}/sankey_all_combinations.html")
        
        # Create individual charts for each combination
        created_count = 0
        for combo_key, result in results_by_combo.items():
            if not result['paths'] or result['sessions_with_path'] == 0:
                continue
            
            # Only create charts for combinations with meaningful conversion
            if result['conversion_rate'] < 0.1:  # Skip if less than 0.1% conversion
                continue
                
            print(f"Creating Sankey diagram for {combo_key}...")
            
            combo_color = combo_colors.get(combo_key, 'rgba(128, 128, 128, 0.8)')
            
            # Build node and link structure for this combination
            nodes_dict = {}
            links_list = []
            
            # Get top paths for this combination
            top_paths = result['paths'][:top_n_paths]
            
            for path, count in top_paths:
                # Create nodes for each screen in the path
                for i, screen in enumerate(path):
                    node_key = f"{screen}_{i}"
                    
                    if node_key not in nodes_dict:
                        # Determine node color
                        if screen == start_screen and i == 0:
                            color = 'rgba(46, 125, 50, 0.9)'  # Green for start
                        elif screen == end_screen:
                            color = 'rgba(198, 40, 40, 0.9)'  # Red for end
                        else:
                            # Use combination color with varying intensity
                            intensity_factor = 0.6 + (0.4 * i / max(len(path) - 1, 1))
                            color = combo_color.replace('0.8', str(intensity_factor))
                        
                        nodes_dict[node_key] = {
                            'label': screen,
                            'color': color,
                            'position': i
                        }
                
                # Create links between consecutive nodes
                for i in range(len(path) - 1):
                    source_key = f"{path[i]}_{i}"
                    target_key = f"{path[i+1]}_{i+1}"
                    
                    # Check if link exists and update count
                    link_found = False
                    for link in links_list:
                        if link['source_key'] == source_key and link['target_key'] == target_key:
                            link['value'] += count
                            link_found = True
                            break
                    
                    if not link_found:
                        links_list.append({
                            'source_key': source_key,
                            'target_key': target_key,
                            'value': count
                        })
            
            # Sort nodes by position
            sorted_nodes = sorted(nodes_dict.items(), key=lambda x: (x[1]['position'], x[0]))
            
            # Create node list and mapping
            node_list = []
            node_to_idx = {}
            for node_key, node_info in sorted_nodes:
                node_to_idx[node_key] = len(node_list)
                node_list.append(node_info)
            
            # Calculate x positions
            max_position = max(node['position'] for node in node_list) if node_list else 0
            x_positions = []
            for node in node_list:
                x_pos = node['position'] / max_position if max_position > 0 else 0.5
                x_positions.append(x_pos)
            
            # Convert links to use indices
            final_links = []
            for link in links_list:
                if link['source_key'] in node_to_idx and link['target_key'] in node_to_idx:
                    final_links.append({
                        'source': node_to_idx[link['source_key']],
                        'target': node_to_idx[link['target_key']],
                        'value': link['value']
                    })
            
            # Create the Sankey diagram for this combination
            fig = go.Figure(data=[go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="white", width=2),
                    label=[node['label'] for node in node_list],
                    color=[node['color'] for node in node_list],
                    x=x_positions
                ),
                link=dict(
                    source=[link['source'] for link in final_links],
                    target=[link['target'] for link in final_links],
                    value=[link['value'] for link in final_links],
                    color=combo_color.replace('0.8', '0.3'),
                    hovertemplate='<b>%{source.label} → %{target.label}</b><br>Sessions: %{value}<extra></extra>'
                ),
                textfont=dict(size=12, color='black')
            )])
            
            # Update layout with statistics
            conversion_rate = result['conversion_rate']
            total_sessions = result['total_sessions']
            sessions_with_path = result['sessions_with_path']
            
            # Parse combination key for display
            combo_display = combo_key.replace(' & ', ', ').replace('=', ': ')
            
            fig.update_layout(
                title={
                    'text': f"User Flow: {start_screen} → {end_screen}<br>" +
                           f"<span style='font-size:14px'>{combo_display} | " +
                           f"Conversion Rate: {conversion_rate:.1f}% ({sessions_with_path}/{total_sessions} sessions)</span>",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': 'black'}
                },
                font=dict(size=12),
                height=500,
                width=1400,
                margin=dict(l=10, r=10, t=100, b=40),
                paper_bgcolor='rgba(250, 250, 250, 1)',
                plot_bgcolor='rgba(255, 255, 255, 0)'
            )
            
            # Save the chart with safe filename
            safe_combo_name = combo_key.replace(' & ', '_').replace('=', '_').replace(' ', '_').replace('/', '_')
            output_file = f"{output_dir}/sankey_{safe_combo_name}.html"
            fig.write_html(output_file)
            print(f"  Saved to {output_file}")
            created_count += 1
        
        print(f"\nCreated {created_count + 1} Sankey diagrams in {output_dir}/")
    
    def _create_combined_sankey_multiple_fields(self, all_paths, start_screen, end_screen,
                                               field_names, combo_colors, output_file):
        """Create a combined Sankey diagram showing all combinations of multiple fields together."""
        
        # Aggregate paths across all combinations
        path_counts = {}
        for path, count, combo in all_paths:
            path_str = ' -> '.join(path)
            if path_str not in path_counts:
                path_counts[path_str] = 0
            path_counts[path_str] += count
        
        # Sort by count and take top paths
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Build nodes and links
        nodes_dict = {}
        links_list = []
        
        for path_str, total_count in sorted_paths:
            path = path_str.split(' -> ')
            
            for i, screen in enumerate(path):
                node_key = f"{screen}_{i}"
                
                if node_key not in nodes_dict:
                    # Determine node color
                    if screen == start_screen and i == 0:
                        color = 'rgba(46, 125, 50, 0.9)'  # Green for start
                    elif screen == end_screen:
                        color = 'rgba(198, 40, 40, 0.9)'  # Red for end
                    else:
                        color = 'rgba(100, 150, 200, 0.8)'  # Blue for middle nodes
                    
                    nodes_dict[node_key] = {
                        'label': screen,
                        'color': color,
                        'position': i
                    }
            
            # Create links
            for i in range(len(path) - 1):
                source_key = f"{path[i]}_{i}"
                target_key = f"{path[i+1]}_{i+1}"
                
                link_found = False
                for link in links_list:
                    if link['source_key'] == source_key and link['target_key'] == target_key:
                        link['value'] += total_count
                        link_found = True
                        break
                
                if not link_found:
                    links_list.append({
                        'source_key': source_key,
                        'target_key': target_key,
                        'value': total_count
                    })
        
        # Sort and create node list
        sorted_nodes = sorted(nodes_dict.items(), key=lambda x: (x[1]['position'], x[0]))
        node_list = []
        node_to_idx = {}
        
        for node_key, node_info in sorted_nodes:
            node_to_idx[node_key] = len(node_list)
            node_list.append(node_info)
        
        # Calculate positions
        max_position = max(node['position'] for node in node_list) if node_list else 0
        x_positions = [node['position'] / max_position if max_position > 0 else 0.5 for node in node_list]
        
        # Convert links to indices
        final_links = []
        for link in links_list:
            if link['source_key'] in node_to_idx and link['target_key'] in node_to_idx:
                final_links.append({
                    'source': node_to_idx[link['source_key']],
                    'target': node_to_idx[link['target_key']],
                    'value': link['value']
                })
        
        # Create the combined Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="white", width=2),
                label=[node['label'] for node in node_list],
                color=[node['color'] for node in node_list],
                x=x_positions
            ),
            link=dict(
                source=[link['source'] for link in final_links],
                target=[link['target'] for link in final_links],
                value=[link['value'] for link in final_links],
                color='rgba(150, 150, 150, 0.3)',
                hovertemplate='<b>%{source.label} → %{target.label}</b><br>Total Sessions: %{value}<extra></extra>'
            ),
            textfont=dict(size=12, color='black')
        )])
        
        # Calculate total statistics
        total_sessions = sum(count for _, count, _ in all_paths)
        fields_display = ' + '.join(field_names)
        
        fig.update_layout(
            title={
                'text': f"User Flow: {start_screen} → {end_screen}<br>" +
                       f"<span style='font-size:14px'>All Combinations of {fields_display} | Total Sessions: {total_sessions}</span>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'black'}
            },
            font=dict(size=12),
            height=600,
            width=1400,
            margin=dict(l=10, r=10, t=100, b=40),
            paper_bgcolor='rgba(250, 250, 250, 1)',
            plot_bgcolor='rgba(255, 255, 255, 0)'
        )
        
        fig.write_html(output_file)
        print(f"  Saved combined view to {output_file}")
    
    def _generate_color_palette(self, num_colors: int) -> List[str]:
        """Generate a color palette with the specified number of colors."""
        import colorsys
        
        colors = []
        for i in range(num_colors):
            # Use HSV color space for better distribution
            hue = i / num_colors
            saturation = 0.7
            value = 0.8
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to rgba string
            r, g, b = [int(x * 255) for x in rgb]
            colors.append(f'rgba({r}, {g}, {b}, 0.8)')
        
        return colors
    
    def _create_combined_sankey_field(self, all_paths, start_screen, end_screen, 
                                     field_name, value_colors, output_file):
        """Create a combined Sankey diagram showing all field values together."""
        from sankey_unified import create_unified_sankey_data
        
        # Use unified sankey data creation
        sankey_data = create_unified_sankey_data(all_paths, start_screen, end_screen)
        
        # Create the Sankey diagram with unified nodes
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="white", width=2),
                label=[node['label'] for node in sankey_data['nodes']],
                color=[node['color'] for node in sankey_data['nodes']],
                x=sankey_data['x_positions']
            ),
            link=dict(
                source=[link['source'] for link in sankey_data['links']],
                target=[link['target'] for link in sankey_data['links']],
                value=[link['value'] for link in sankey_data['links']],
                color='rgba(150, 150, 150, 0.3)',
                hovertemplate='<b>%{source.label} → %{target.label}</b><br>Total Sessions: %{value}<extra></extra>'
            ),
            textfont=dict(size=12, color='black')
        )])
        
        # Calculate total statistics
        total_sessions = sum(count for _, count, _ in all_paths)
        
        fig.update_layout(
            title={
                'text': f"User Flow: {start_screen} → {end_screen}<br>" +
                       f"<span style='font-size:14px'>All {field_name} Values Combined | Total Sessions: {total_sessions}</span>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'black'}
            },
            font=dict(size=12),
            height=600,
            width=1400,
            margin=dict(l=10, r=10, t=100, b=40),
            paper_bgcolor='rgba(250, 250, 250, 1)',
            plot_bgcolor='rgba(255, 255, 255, 0)'
        )
        
        fig.write_html(output_file)
        print(f"  Saved combined view to {output_file}")
        return
    
    def _create_combined_sankey_field_OLD(self, all_paths, start_screen, end_screen, 
                                     field_name, value_colors, output_file):
        """OLD VERSION - Create a combined Sankey diagram showing all field values together."""
        # OLD Implementation follows...
        sorted_paths = []
        
        # Determine if we have context (paths with different starting positions for start_screen)
        has_context = False
        max_context_length = 0
        for path_str, _ in sorted_paths:
            path = path_str.split(' -> ')
            if start_screen in path:
                start_idx = path.index(start_screen)
                if start_idx > 0:
                    has_context = True
                    max_context_length = max(max_context_length, start_idx)
        
        # Build nodes and links with unified positions
        nodes_dict = {}
        links_list = []
        
        for path_str, total_count in sorted_paths:
            path = path_str.split(' -> ')
            
            # Find where the main path starts
            start_idx = path.index(start_screen) if start_screen in path else 0
            
            for i, screen in enumerate(path):
                # Calculate unified position
                if has_context:
                    # Normalize position relative to start_screen
                    if i < start_idx:
                        # Context node - use negative positions
                        unified_pos = i - start_idx
                        node_key = f"{screen}_context_{unified_pos}"
                    else:
                        # Main path node - use positive positions
                        unified_pos = i - start_idx
                        node_key = f"{screen}_main_{unified_pos}"
                else:
                    # No context, use simple positioning
                    node_key = f"{screen}_{i}"
                    unified_pos = i
                
                if node_key not in nodes_dict:
                    # Determine node color and label
                    if i < start_idx and has_context:
                        # Context node
                        color = 'rgba(150, 150, 200, 0.7)'
                        label = f"{screen} (context)"
                    elif screen == start_screen:
                        color = 'rgba(46, 125, 50, 0.9)'  # Green for start
                        label = f"{screen} (START)"
                    elif screen == end_screen:
                        color = 'rgba(198, 40, 40, 0.9)'  # Red for end
                        label = f"{screen} (END)"
                    else:
                        color = 'rgba(100, 150, 200, 0.8)'  # Blue for middle nodes
                        label = screen
                    
                    nodes_dict[node_key] = {
                        'label': label,
                        'color': color,
                        'position': unified_pos + max_context_length if has_context else i,
                        'node_key': node_key
                    }
            
            # Create links
            for i in range(len(path) - 1):
                # Determine source node key
                if has_context:
                    if i < start_idx:
                        source_unified_pos = i - start_idx
                        source_key = f"{path[i]}_context_{source_unified_pos}"
                    else:
                        source_unified_pos = i - start_idx
                        source_key = f"{path[i]}_main_{source_unified_pos}"
                else:
                    source_key = f"{path[i]}_{i}"
                
                # Determine target node key
                if has_context:
                    if i + 1 < start_idx:
                        target_unified_pos = (i + 1) - start_idx
                        target_key = f"{path[i+1]}_context_{target_unified_pos}"
                    else:
                        target_unified_pos = (i + 1) - start_idx
                        target_key = f"{path[i+1]}_main_{target_unified_pos}"
                else:
                    target_key = f"{path[i+1]}_{i+1}"
                
                link_found = False
                for link in links_list:
                    if link['source_key'] == source_key and link['target_key'] == target_key:
                        link['value'] += total_count
                        link_found = True
                        break
                
                if not link_found:
                    links_list.append({
                        'source_key': source_key,
                        'target_key': target_key,
                        'value': total_count
                    })
        
        # Sort and create node list
        sorted_nodes = sorted(nodes_dict.items(), key=lambda x: (x[1]['position'], x[0]))
        node_list = []
        node_to_idx = {}
        
        for node_key, node_info in sorted_nodes:
            node_to_idx[node_key] = len(node_list)
            node_list.append(node_info)
        
        # Calculate positions
        max_position = max(node['position'] for node in node_list) if node_list else 0
        x_positions = [node['position'] / max_position if max_position > 0 else 0.5 for node in node_list]
        
        # Convert links to indices
        final_links = []
        for link in links_list:
            if link['source_key'] in node_to_idx and link['target_key'] in node_to_idx:
                final_links.append({
                    'source': node_to_idx[link['source_key']],
                    'target': node_to_idx[link['target_key']],
                    'value': link['value']
                })
        
        # Create the combined Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="white", width=2),
                label=[node['label'] for node in node_list],
                color=[node['color'] for node in node_list],
                x=x_positions
            ),
            link=dict(
                source=[link['source'] for link in final_links],
                target=[link['target'] for link in final_links],
                value=[link['value'] for link in final_links],
                color='rgba(150, 150, 150, 0.3)',
                hovertemplate='<b>%{source.label} → %{target.label}</b><br>Total Sessions: %{value}<extra></extra>'
            ),
            textfont=dict(size=12, color='black')
        )])
        
        # Calculate total statistics
        total_sessions = sum(count for _, count, _ in all_paths)
        
        fig.update_layout(
            title={
                'text': f"User Flow: {start_screen} → {end_screen}<br>" +
                       f"<span style='font-size:14px'>All {field_name} Values Combined | Total Sessions: {total_sessions}</span>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'black'}
            },
            font=dict(size=12),
            height=600,
            width=1400,
            margin=dict(l=10, r=10, t=100, b=40),
            paper_bgcolor='rgba(250, 250, 250, 1)',
            plot_bgcolor='rgba(255, 255, 255, 0)'
        )
        
        fig.write_html(output_file)
        print(f"  Saved combined view to {output_file}")
    
    def _create_combined_sankey(self, all_paths, start_screen, end_screen, source_colors, output_file):
        """Create a combined Sankey diagram showing all traffic sources together."""
        
        # Aggregate paths across all sources
        path_counts = {}
        for path, count, source in all_paths:
            path_str = ' -> '.join(path)
            if path_str not in path_counts:
                path_counts[path_str] = 0
            path_counts[path_str] += count
        
        # Sort by count and take top paths
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Build nodes and links
        nodes_dict = {}
        links_list = []
        
        for path_str, total_count in sorted_paths:
            path = path_str.split(' -> ')
            
            for i, screen in enumerate(path):
                node_key = f"{screen}_{i}"
                
                if node_key not in nodes_dict:
                    # Determine node color
                    if screen == start_screen and i == 0:
                        color = 'rgba(46, 125, 50, 0.9)'  # Green for start
                    elif screen == end_screen:
                        color = 'rgba(198, 40, 40, 0.9)'  # Red for end
                    else:
                        color = 'rgba(100, 150, 200, 0.8)'  # Blue for middle nodes
                    
                    nodes_dict[node_key] = {
                        'label': screen,
                        'color': color,
                        'position': i
                    }
            
            # Create links
            for i in range(len(path) - 1):
                source_key = f"{path[i]}_{i}"
                target_key = f"{path[i+1]}_{i+1}"
                
                link_found = False
                for link in links_list:
                    if link['source_key'] == source_key and link['target_key'] == target_key:
                        link['value'] += total_count
                        link_found = True
                        break
                
                if not link_found:
                    links_list.append({
                        'source_key': source_key,
                        'target_key': target_key,
                        'value': total_count
                    })
        
        # Sort and create node list
        sorted_nodes = sorted(nodes_dict.items(), key=lambda x: (x[1]['position'], x[0]))
        node_list = []
        node_to_idx = {}
        
        for node_key, node_info in sorted_nodes:
            node_to_idx[node_key] = len(node_list)
            node_list.append(node_info)
        
        # Calculate positions
        max_position = max(node['position'] for node in node_list) if node_list else 0
        x_positions = [node['position'] / max_position if max_position > 0 else 0.5 for node in node_list]
        
        # Convert links to indices
        final_links = []
        for link in links_list:
            if link['source_key'] in node_to_idx and link['target_key'] in node_to_idx:
                final_links.append({
                    'source': node_to_idx[link['source_key']],
                    'target': node_to_idx[link['target_key']],
                    'value': link['value']
                })
        
        # Create the combined Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="white", width=2),
                label=[node['label'] for node in node_list],
                color=[node['color'] for node in node_list],
                x=x_positions
            ),
            link=dict(
                source=[link['source'] for link in final_links],
                target=[link['target'] for link in final_links],
                value=[link['value'] for link in final_links],
                color='rgba(150, 150, 150, 0.3)',
                hovertemplate='<b>%{source.label} → %{target.label}</b><br>Total Sessions: %{value}<extra></extra>'
            ),
            textfont=dict(size=12, color='black')
        )])
        
        # Calculate total statistics
        total_sessions = sum(count for _, count, _ in all_paths)
        
        fig.update_layout(
            title={
                'text': f"User Flow: {start_screen} → {end_screen}<br>" +
                       f"<span style='font-size:14px'>All Traffic Sources Combined | Total Sessions: {total_sessions}</span>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'black'}
            },
            font=dict(size=12),
            height=600,
            width=1400,
            margin=dict(l=10, r=10, t=100, b=40),
            paper_bgcolor='rgba(250, 250, 250, 1)',
            plot_bgcolor='rgba(255, 255, 255, 0)'
        )
        
        fig.write_html(output_file)
        print(f"  Saved combined view to {output_file}")
    
    def create_sankey_by_traffic_source(self, start_screen: str, end_screen: str,
                                       top_n_paths: int = 10, output_file: Optional[str] = None):
        """Create a Sankey diagram split by traffic source with color coding."""
        
        # Get paths segmented by traffic source
        results_by_source = self.analyzer.find_paths_by_traffic_source(
            start_screen, end_screen, max_length=10
        )
        
        if not results_by_source:
            print(f"No paths found from '{start_screen}' to '{end_screen}'")
            return
        
        # Define color palette for traffic sources
        source_colors = {
            'google_organic': 'rgba(66, 133, 244, 0.8)',  # Google blue
            'google_ads': 'rgba(234, 67, 53, 0.8)',       # Google red
            'facebook_campaign': 'rgba(24, 119, 242, 0.8)', # Facebook blue
            'social_organic': 'rgba(29, 161, 242, 0.8)',   # Twitter blue
            'direct': 'rgba(52, 168, 83, 0.8)',            # Green
            'email_newsletter': 'rgba(251, 188, 5, 0.8)',  # Yellow
            'affiliate': 'rgba(154, 51, 205, 0.8)',        # Purple
            'default': 'rgba(128, 128, 128, 0.8)'          # Gray
        }
        
        # Build comprehensive node and link structure
        all_nodes = {}
        all_links = []
        node_list = []
        node_to_idx = {}
        
        # Process each traffic source
        for source, result in results_by_source.items():
            if not result['paths']:
                continue
                
            source_color = source_colors.get(source, source_colors['default'])
            
            # Get top paths for this source
            top_paths = result['paths'][:top_n_paths]
            
            for path, count in top_paths:
                # Create unique nodes for each source-path combination
                for i, screen in enumerate(path):
                    # Create a unique node key that includes source
                    node_key = f"{screen}_{source}_{i}"
                    
                    if node_key not in all_nodes:
                        # Determine node properties
                        if screen == start_screen and i == 0:
                            label = f"{screen}\n[{source}]"
                            color = source_color
                        elif screen == end_screen:
                            label = screen
                            color = 'rgba(220, 20, 60, 0.9)'  # Red for end
                        else:
                            label = screen
                            # Blend source color with position
                            color = source_color
                        
                        all_nodes[node_key] = {
                            'label': label,
                            'color': color,
                            'position': i,
                            'source': source,
                            'screen': screen
                        }
                
                # Create links between consecutive nodes
                for i in range(len(path) - 1):
                    source_key = f"{path[i]}_{source}_{i}"
                    target_key = f"{path[i+1]}_{source}_{i+1}"
                    
                    # Find existing link or create new one
                    link_found = False
                    for link in all_links:
                        if link['source_key'] == source_key and link['target_key'] == target_key:
                            link['value'] += count
                            link_found = True
                            break
                    
                    if not link_found:
                        all_links.append({
                            'source_key': source_key,
                            'target_key': target_key,
                            'value': count,
                            'color': source_color.replace('0.8', '0.4'),  # Lighter version
                            'traffic_source': source
                        })
        
        # Sort nodes by position and source for better layout
        sorted_nodes = sorted(all_nodes.items(), key=lambda x: (x[1]['position'], x[1]['source']))
        
        # Create node list and mapping
        for node_key, node_info in sorted_nodes:
            node_to_idx[node_key] = len(node_list)
            node_list.append(node_info)
        
        # Calculate x positions based on screen position in path
        max_position = max(node['position'] for node in node_list)
        x_positions = []
        y_positions = []
        
        # Group nodes by position
        position_groups = defaultdict(list)
        for idx, node in enumerate(node_list):
            position_groups[node['position']].append(idx)
        
        # Assign x and y positions
        for idx, node in enumerate(node_list):
            x_pos = node['position'] / max_position if max_position > 0 else 0.5
            x_positions.append(x_pos)
            
            # Spread nodes vertically at each position
            nodes_at_pos = position_groups[node['position']]
            node_index_at_pos = nodes_at_pos.index(idx)
            num_nodes_at_pos = len(nodes_at_pos)
            
            if num_nodes_at_pos == 1:
                y_pos = 0.5
            else:
                y_pos = 0.1 + (0.8 * node_index_at_pos / (num_nodes_at_pos - 1))
            y_positions.append(y_pos)
        
        # Convert links to use indices
        final_links = []
        for link in all_links:
            if link['source_key'] in node_to_idx and link['target_key'] in node_to_idx:
                final_links.append({
                    'source': node_to_idx[link['source_key']],
                    'target': node_to_idx[link['target_key']],
                    'value': link['value'],
                    'color': link['color'],
                    'label': f"{link['traffic_source']}: {link['value']}"
                })
        
        # Create the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="white", width=2),
                label=[node['label'] for node in node_list],
                color=[node['color'] for node in node_list],
                x=x_positions,
                y=y_positions,
                hovertemplate='<b>%{label}</b><br>Traffic Source: %{customdata}<extra></extra>',
                customdata=[node['source'] for node in node_list]
            ),
            link=dict(
                source=[link['source'] for link in final_links],
                target=[link['target'] for link in final_links],
                value=[link['value'] for link in final_links],
                color=[link['color'] for link in final_links],
                hovertemplate='<b>%{source.label} → %{target.label}</b><br>%{customdata}<br>Flow count: %{value}<extra></extra>',
                customdata=[link['label'] for link in final_links]
            ),
            textfont=dict(size=10, color='black')
        )])
        
        # Create legend for traffic sources
        legend_annotations = []
        y_pos = 0.98
        for i, (source, color) in enumerate(source_colors.items()):
            if source == 'default':
                continue
            if source in [r[0] for r in results_by_source.items() if r[1]['paths']]:
                legend_annotations.append(
                    dict(
                        x=1.02,
                        y=y_pos - (i * 0.04),
                        xref='paper',
                        yref='paper',
                        text=f'<span style="color:{color}">●</span> {source.replace("_", " ").title()}',
                        showarrow=False,
                        font=dict(size=11),
                        xanchor='left'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"User Flow by Traffic Source: {start_screen} → {end_screen}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': 'black'}
            },
            annotations=legend_annotations,
            font=dict(size=11),
            height=max(700, len(node_list) * 25),
            width=1800,
            margin=dict(l=10, r=200, t=80, b=40),
            paper_bgcolor='rgba(250, 250, 250, 1)',
            plot_bgcolor='rgba(255, 255, 255, 0)'
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Traffic source Sankey diagram saved to {output_file}")
        else:
            fig.show()
    
    def create_simulation_comparison_chart(self, comparison_results: dict, 
                                          output_file: Optional[str] = None):
        """Create comparison chart for Monte Carlo simulation results.
        
        Args:
            comparison_results: Results from MonteCarloSimulator.compare_scenarios()
            output_file: Optional file path to save the chart
        """
        conversion_changes = comparison_results['conversion_changes']
        
        if not conversion_changes:
            print("No conversion data available for comparison")
            return
        
        # Prepare data for plotting
        goals = list(conversion_changes.keys())
        baseline_rates = [conversion_changes[goal]['baseline'] for goal in goals]
        modified_rates = [conversion_changes[goal]['modified'] for goal in goals]
        
        # Create subplot with bar chart
        fig = go.Figure()
        
        # Add baseline bars
        fig.add_trace(go.Bar(
            name='Baseline',
            x=goals,
            y=baseline_rates,
            marker_color='lightblue',
            text=[f"{rate:.1f}%" for rate in baseline_rates],
            textposition='auto'
        ))
        
        # Add modified bars
        fig.add_trace(go.Bar(
            name='Modified Scenario',
            x=goals,
            y=modified_rates,
            marker_color='darkblue',
            text=[f"{rate:.1f}%" for rate in modified_rates],
            textposition='auto'
        ))
        
        # Add change annotations
        for i, goal in enumerate(goals):
            change = conversion_changes[goal]['percent_change']
            color = 'green' if change > 0 else 'red'
            fig.add_annotation(
                x=goal,
                y=max(baseline_rates[i], modified_rates[i]) + 2,
                text=f"{change:+.1f}%",
                showarrow=False,
                font=dict(color=color, size=12, family="Arial Black")
            )
        
        # Update layout
        fig.update_layout(
            title=f"Monte Carlo Simulation: Conversion Rate Comparison<br><sub>Based on {comparison_results['n_simulations']:,} simulations each</sub>",
            xaxis_title="Goal Screens",
            yaxis_title="Conversion Rate (%)",
            barmode='group',
            height=500,
            showlegend=True
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Simulation comparison chart saved to {output_file}")
        else:
            fig.show()
    
    def export_to_graphviz(self, output_file: str, start_screen: Optional[str] = None,
                          end_screen: Optional[str] = None):
        graph = pydot.Dot(graph_type='digraph', rankdir='LR')
        
        if start_screen and end_screen:
            paths = self.analyzer.find_paths(start_screen, end_screen)[:10]
            edges_to_add = set()
            nodes_to_add = set()
            
            for path, count in paths:
                for i in range(len(path) - 1):
                    edges_to_add.add((path[i], path[i+1]))
                    nodes_to_add.add(path[i])
                    nodes_to_add.add(path[i+1])
            
            for node in nodes_to_add:
                color = 'green' if node == start_screen else 'red' if node == end_screen else 'lightblue'
                graph.add_node(pydot.Node(node, style="filled", fillcolor=color))
            
            for from_node, to_node in edges_to_add:
                weight = self.analyzer.screen_transitions[from_node][to_node]
                graph.add_edge(pydot.Edge(from_node, to_node, label=str(weight)))
        else:
            for from_screen, transitions in self.analyzer.screen_transitions.items():
                for to_screen, count in transitions.items():
                    graph.add_edge(pydot.Edge(from_screen, to_screen, label=str(count)))
        
        graph.write_png(output_file)
        print(f"Graphviz diagram saved to {output_file}")