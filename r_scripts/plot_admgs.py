#!/usr/bin/env python3
"""
Script to plot ADMGs from saved matrix files.
Creates proper graph visualizations showing directed and bidirected edges.
"""

import os
import numpy as np
from pathlib import Path
import glob
import pydot
import io
from PIL import Image

def parse_matrix_file(filepath):
    """Parse a matrix file and return the adjacency matrix."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    if len(lines) < 2:
        return None, None
    
    # First line contains column names
    node_names = lines[0].split('\t')
    
    # Parse matrix data
    matrix_data = []
    for line in lines[1:]:
        row = line.split('\t')
        # Skip the first element (row name) and convert to integers
        matrix_data.append([int(x) for x in row[1:]])
    
    return np.array(matrix_data), node_names

# Removed matplotlib/NetworkX functions - using only pydot

def create_pydot_graph(matrix, node_names):
    """Create a pydot graph from ADMG adjacency matrix with proper 101 handling."""
    graph = pydot.Dot(graph_type='digraph')
    
    # Add nodes
    for name in node_names:
        node = pydot.Node(name, style='filled', fillcolor='lightblue', 
                         fontname='Arial', fontsize='16', fontweight='bold')
        graph.add_node(node)
    
    # Add edges based on matrix values
    n = len(node_names)
    processed_bidirected = set()  # Track processed bidirected edges to avoid duplicates
    
    for i in range(n):
        for j in range(n):
            if matrix[i, j] == 1:  # Directed edge only
                edge = pydot.Edge(node_names[i], node_names[j], 
                                color='black', arrowhead='normal', penwidth=2)
                graph.add_edge(edge)
            elif matrix[i, j] == 100:  # Bidirected edge only
                # Only add bidirected edge once (avoid duplicates)
                edge_key = tuple(sorted([node_names[i], node_names[j]]))
                if edge_key not in processed_bidirected:
                    edge = pydot.Edge(node_names[i], node_names[j], 
                                    color='black', arrowhead='normal', 
                                    arrowtail='normal', dir='both', penwidth=2)
                    graph.add_edge(edge)
                    processed_bidirected.add(edge_key)
            elif matrix[i, j] == 101:  # Treat as directed edge (bidirected handled by 100 on other side)
                edge = pydot.Edge(node_names[i], node_names[j], 
                                color='black', arrowhead='normal', penwidth=2)
                graph.add_edge(edge)
    
    return graph

def plot_admg_pydot(matrix, node_names, title, output_path):
    """Plot an ADMG using pydot for better edge handling."""
    try:
        # Create pydot graph
        graph = create_pydot_graph(matrix, node_names)
        
        # Set graph attributes
        graph.set_graph_defaults(rankdir='TB', size='10,8', dpi='300')
        
        # Generate PNG data
        png_data = graph.create_png()
        
        # Convert to PIL Image and save
        img = Image.open(io.BytesIO(png_data))
        
        # Add title as text overlay
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fallback to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        # Get text size and position it at the top
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (img.width - text_width) // 2
        text_y = 10
        
        # Draw title
        draw.text((text_x, text_y), title, fill='black', font=font)
        
        # No legend needed - edge types are self-explanatory
        
        # Save the image
        img.save(output_path, 'PNG', dpi=(300, 300))
        
        print(f"Saved (pydot): {output_path}")
        return True
        
    except Exception as e:
        print(f"Pydot plotting failed: {e}")
        return False

# Removed matplotlib plotting function - using only pydot

def plot_all_admgs(plots_dir="plots"):
    """Plot all ADMG matrix files."""
    plots_path = Path(plots_dir)
    
    if not plots_path.exists():
        print(f"Directory {plots_dir} does not exist!")
        return
    
    # Find all matrix files
    matrix_files = list(plots_path.glob("*_matrix.txt"))
    
    if not matrix_files:
        print("No matrix files found!")
        return
    
    print(f"Found {len(matrix_files)} matrix files to plot")
    
    # Create output directory for Python plots
    python_plots_dir = plots_path / "python_plots"
    python_plots_dir.mkdir(exist_ok=True)
    
    # Separate filtered and unfiltered ADMGs
    filtered_files = [f for f in matrix_files if '_all_' not in f.name]
    unfiltered_files = [f for f in matrix_files if '_all_' in f.name]
    
    print(f"Filtered ADMGs: {len(filtered_files)}")
    print(f"Unfiltered ADMGs: {len(unfiltered_files)}")
    
    # Plot filtered ADMGs
    for matrix_file in sorted(filtered_files):
        print(f"Processing {matrix_file.name}...")
        
        # Parse matrix
        matrix, node_names = parse_matrix_file(matrix_file)
        
        if matrix is None:
            print(f"Could not parse {matrix_file.name}")
            continue
        
        # Generate title and output path
        base_name = matrix_file.stem.replace('_matrix', '')
        # Extract just the number from the base name (e.g., "pag1_admg_1" -> "1")
        admg_number = base_name.split('_')[-1]
        title = f"ADMG {admg_number}"
        output_path = python_plots_dir / f"{base_name}_python.png"
        
        # Use pydot only
        plot_admg_pydot(matrix, node_names, title, str(output_path))
    
    # Plot unfiltered ADMGs
    for matrix_file in sorted(unfiltered_files):
        print(f"Processing {matrix_file.name}...")
        
        # Parse matrix
        matrix, node_names = parse_matrix_file(matrix_file)
        
        if matrix is None:
            print(f"Could not parse {matrix_file.name}")
            continue
        
        # Generate title and output path
        base_name = matrix_file.stem.replace('_matrix', '')
        # Extract just the number from the base name (e.g., "pag1_admg_1" -> "1")
        admg_number = base_name.split('_')[-1]
        title = f"ADMG {admg_number}"
        output_path = python_plots_dir / f"{base_name}_python.png"
        
        # Use pydot only
        plot_admg_pydot(matrix, node_names, title, str(output_path))
    
    print(f"\nAll plots saved to: {python_plots_dir}")
    
    # Create combined plots for each PAG
    print("\nCreating combined plots...")
    create_combined_pag_plot("pag1", filtered_files, python_plots_dir)
    create_combined_pag_plot("pag2", filtered_files, python_plots_dir)

def create_combined_pag_plot(pag_name, matrix_files, output_dir):
    """Create a combined plot showing all ADMGs for a specific PAG."""
    if not matrix_files:
        return
    
    # Filter files for this specific PAG
    pag_files = [f for f in matrix_files if f.name.startswith(pag_name)]
    if not pag_files:
        return
    
    n_files = len(pag_files)
    if n_files == 0:
        return
    
    # Calculate grid dimensions (3 columns max for readability)
    cols = min(3, n_files)
    rows = (n_files + cols - 1) // cols
    
    # Create a large combined graph
    combined_graph = pydot.Dot(graph_type='digraph')
    combined_graph.set_graph_defaults(rankdir='TB', size='20,15', dpi='300')
    
    # Add subgraph for each ADMG
    for i, matrix_file in enumerate(sorted(pag_files)):
        # Parse matrix
        matrix, node_names = parse_matrix_file(matrix_file)
        if matrix is None:
            continue
        
        # Create subgraph for this ADMG
        base_name = matrix_file.stem.replace('_matrix', '')
        admg_number = base_name.split('_')[-1]
        subgraph_name = f"cluster_{admg_number}"
        
        subgraph = pydot.Subgraph(graph_name=subgraph_name)
        subgraph.set_label(f"ADMG {admg_number}")
        
        # Add nodes to subgraph
        for name in node_names:
            node = pydot.Node(f"{admg_number}_{name}", label=name, 
                             style='filled', fillcolor='lightblue', 
                             fontname='Arial', fontsize='12', fontweight='bold')
            subgraph.add_node(node)
        
        # Add edges to subgraph
        processed_bidirected = set()
        n = len(node_names)
        for row in range(n):
            for col in range(n):
                if matrix[row, col] == 1:  # Directed edge
                    edge = pydot.Edge(f"{admg_number}_{node_names[row]}", 
                                    f"{admg_number}_{node_names[col]}", 
                                    color='black', arrowhead='normal', penwidth=1.5)
                    subgraph.add_edge(edge)
                elif matrix[row, col] == 100:  # Bidirected edge
                    edge_key = tuple(sorted([f"{admg_number}_{node_names[row]}", f"{admg_number}_{node_names[col]}"]))
                    if edge_key not in processed_bidirected:
                        edge = pydot.Edge(f"{admg_number}_{node_names[row]}", 
                                        f"{admg_number}_{node_names[col]}", 
                                        color='black', arrowhead='normal', 
                                        arrowtail='normal', dir='both', penwidth=1.5)
                        subgraph.add_edge(edge)
                        processed_bidirected.add(edge_key)
                elif matrix[row, col] == 101:  # Treat as directed
                    edge = pydot.Edge(f"{admg_number}_{node_names[row]}", 
                                    f"{admg_number}_{node_names[col]}", 
                                    color='black', arrowhead='normal', penwidth=1.5)
                    subgraph.add_edge(edge)
        
        combined_graph.add_subgraph(subgraph)
    
    # Generate PNG data
    try:
        png_data = combined_graph.create_png()
        
        # Convert to PIL Image and save
        img = Image.open(io.BytesIO(png_data))
        
        # Add title
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 28)
            except:
                font = ImageFont.load_default()
        
        title = f"All ADMGs for {pag_name.upper()}"
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (img.width - text_width) // 2
        text_y = 10
        
        draw.text((text_x, text_y), title, fill='black', font=font)
        
        # Save the image
        output_path = output_dir / f"{pag_name}_all_admgs.png"
        img.save(output_path, 'PNG', dpi=(300, 300))
        
        print(f"Combined plot saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Combined plot failed for {pag_name}: {e}")
        return False

if __name__ == "__main__":
    print("ADMG Plotting Script")
    print("===================")
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Plot all ADMGs
    plot_all_admgs()
    
    print("\nDone!")
