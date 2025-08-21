import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import pydot
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class FlowPathVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
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
        matrix = self.analyzer.export_transition_matrix()
        
        if matrix.empty:
            print("No transition data available")
            return
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale='Blues',
            text=matrix.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Screen Transition Heatmap",
            xaxis_title="To Screen",
            yaxis_title="From Screen",
            height=max(400, len(matrix) * 30),
            width=max(600, len(matrix.columns) * 30),
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'}
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Heatmap saved to {output_file}")
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