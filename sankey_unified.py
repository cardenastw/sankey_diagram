"""
Helper module for creating unified Sankey diagrams with context support.
This ensures that the same screen is represented by a single node regardless
of where it appears in different paths.
"""

def create_unified_sankey_data(all_paths, start_screen, end_screen):
    """
    Create unified node and link data for Sankey diagram with context support.
    
    Returns:
        dict with 'nodes' and 'links' for Sankey diagram
    """
    # First pass: identify all unique screens and their roles
    screen_roles = {}
    context_screens = set()
    main_screens = set()
    
    for path, count, _ in all_paths:
        start_idx = path.index(start_screen) if start_screen in path else 0
        
        for i, screen in enumerate(path):
            if i < start_idx:
                context_screens.add(screen)
            else:
                main_screens.add(screen)
    
    # Assign levels to screens based on their typical position
    screen_levels = {}
    level_screens = {}  # level -> list of screens
    
    for path, count, _ in all_paths:
        start_idx = path.index(start_screen) if start_screen in path else 0
        
        for i, screen in enumerate(path):
            # Calculate relative level
            if i < start_idx:
                level = i - start_idx  # Negative for context
            else:
                level = i - start_idx  # 0 or positive for main path
            
            if screen not in screen_levels:
                screen_levels[screen] = level
            else:
                # Keep the most common level for each screen
                screen_levels[screen] = min(screen_levels[screen], level) if screen in context_screens else max(screen_levels[screen], level)
            
            if level not in level_screens:
                level_screens[level] = set()
            level_screens[level].add(screen)
    
    # Create unified nodes
    nodes = []
    node_map = {}  # screen -> node index
    
    # Sort screens by level, then alphabetically
    sorted_screens = sorted(screen_levels.items(), key=lambda x: (x[1], x[0]))
    
    for screen, level in sorted_screens:
        if screen in context_screens and screen not in main_screens:
            label = f"{screen} (context)"
            color = 'rgba(150, 150, 200, 0.7)'
        elif screen == start_screen:
            label = f"{screen} (START)"
            color = 'rgba(46, 125, 50, 0.9)'
        elif screen == end_screen:
            label = f"{screen} (END)"
            color = 'rgba(198, 40, 40, 0.9)'
        else:
            label = screen
            color = 'rgba(100, 150, 200, 0.8)'
        
        node_map[screen] = len(nodes)
        nodes.append({
            'label': label,
            'color': color,
            'level': level
        })
    
    # Create links with proper flow
    link_map = {}  # (source_screen, target_screen) -> value
    
    for path, count, _ in all_paths:
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            key = (source, target)
            
            if key not in link_map:
                link_map[key] = 0
            link_map[key] += count
    
    # Convert to link list
    links = []
    for (source, target), value in link_map.items():
        if source in node_map and target in node_map:
            links.append({
                'source': node_map[source],
                'target': node_map[target],
                'value': value
            })
    
    # Calculate x positions based on levels
    min_level = min(screen_levels.values()) if screen_levels else 0
    max_level = max(screen_levels.values()) if screen_levels else 0
    level_range = max_level - min_level if max_level != min_level else 1
    
    x_positions = []
    for node in nodes:
        x_pos = (node['level'] - min_level) / level_range
        x_positions.append(x_pos)
    
    return {
        'nodes': nodes,
        'links': links,
        'x_positions': x_positions
    }