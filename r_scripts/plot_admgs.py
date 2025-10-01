#!/usr/bin/env python3
"""
Script to plot ADMGs from saved matrix files.
Creates proper graph visualizations showing directed and bidirected edges.
"""

import os
import argparse
import numpy as np
from pathlib import Path
import pydot
import io
from PIL import Image
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Configuration for ADMG plotting."""
    default_plots_dir: str = "plots"
    max_cols_per_row: int = 3
    node_font_size: int = 16
    node_font_size_combined: int = 12
    title_font_size: int = 24
    title_font_size_combined: int = 28
    image_dpi: int = 300
    graph_size: str = "10,8"
    combined_graph_size: str = "20,15"
    node_color: str = "lightblue"
    edge_color: str = "black"
    edge_width: float = 2.0
    edge_width_combined: float = 1.5


class MatrixParser:
    """Handles parsing of ADMG matrix files."""
    
    @staticmethod
    def parse_matrix_file(filepath: Path) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Parse a matrix file and return the adjacency matrix and node names."""
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            # Skip empty lines
            lines = [line.strip() for line in lines if line.strip()]

            if len(lines) < 2:
                return None, None

            # First line contains column names
            node_names = lines[0].split("\t")

            # Parse matrix data
            matrix_data = []
            for line in lines[1:]:
                row = line.split("\t")
                # Skip the first element (row name) and convert to integers
                matrix_data.append([int(x) for x in row[1:]])

            return np.array(matrix_data), node_names
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None, None


class ADMGPlotter:
    """Handles plotting of ADMGs using pydot."""
    
    def __init__(self, config: PlotConfig):
        self.config = config
    
    def create_pydot_graph(self, matrix: np.ndarray, node_names: List[str]) -> pydot.Dot:
        """Create a pydot graph from ADMG adjacency matrix with proper 101 handling."""
        graph = pydot.Dot(graph_type="digraph")

        # Add nodes
        for name in node_names:
            node = pydot.Node(
                name,
                style="filled",
                fillcolor=self.config.node_color,
                fontname="Arial",
                fontsize=str(self.config.node_font_size),
                fontweight="bold",
            )
            graph.add_node(node)

        # Add edges based on matrix values
        n = len(node_names)
        processed_bidirected = set()  # Track processed bidirected edges to avoid duplicates

        for i in range(n):
            for j in range(n):
                if matrix[i, j] == 1:  # Directed edge only
                    edge = pydot.Edge(
                        node_names[i],
                        node_names[j],
                        color=self.config.edge_color,
                        arrowhead="normal",
                        penwidth=self.config.edge_width,
                    )
                    graph.add_edge(edge)
                elif matrix[i, j] == 100:  # Bidirected edge only
                    # Only add bidirected edge once (avoid duplicates)
                    edge_key = tuple(sorted([node_names[i], node_names[j]]))
                    if edge_key not in processed_bidirected:
                        edge = pydot.Edge(
                            node_names[i],
                            node_names[j],
                            color=self.config.edge_color,
                            arrowhead="normal",
                            arrowtail="normal",
                            dir="both",
                            penwidth=self.config.edge_width,
                        )
                        graph.add_edge(edge)
                        processed_bidirected.add(edge_key)
                elif (
                    matrix[i, j] == 101
                ):  # Treat as directed edge (bidirected handled by 100 on other side)
                    edge = pydot.Edge(
                        node_names[i],
                        node_names[j],
                        color=self.config.edge_color,
                        arrowhead="normal",
                        penwidth=self.config.edge_width,
                    )
                    graph.add_edge(edge)

        return graph

    def plot_admg(self, matrix: np.ndarray, node_names: List[str], title: str, output_path: str) -> bool:
        """Plot an ADMG using pydot for better edge handling."""
        try:
            # Create pydot graph
            graph = self.create_pydot_graph(matrix, node_names)

            # Set graph attributes
            graph.set_graph_defaults(rankdir="TB", size=self.config.graph_size, dpi=str(self.config.image_dpi))

            # Generate PNG data
            png_data = graph.create_png()

            # Convert to PIL Image and save
            img = Image.open(io.BytesIO(png_data))

            # Add title as text overlay
            self._add_title_to_image(img, title, self.config.title_font_size)

            # Save the image
            img.save(output_path, "PNG", dpi=(self.config.image_dpi, self.config.image_dpi))

            print(f"Saved (pydot): {output_path}")
            return True

        except Exception as e:
            print(f"Pydot plotting failed: {e}")
            return False

    def _add_title_to_image(self, img: Image.Image, title: str, font_size: int) -> None:
        """Add title text to the image."""
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)

        # Try to use a nice font, fallback to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        # Get text size and position it at the top
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (img.width - text_width) // 2
        text_y = 10

        # Draw title
        draw.text((text_x, text_y), title, fill="black", font=font)


class ADMGFileHandler:
    """Handles file operations for ADMG plotting."""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        self.parser = MatrixParser()
    
    def find_matrix_files(self, plots_dir: str) -> Tuple[List[Path], List[Path]]:
        """Find and categorize matrix files."""
        plots_path = Path(plots_dir)
        
        if not plots_path.exists():
            print(f"Directory {plots_dir} does not exist!")
            return [], []

        # Find all matrix files
        matrix_files = list(plots_path.glob("*_matrix.txt"))
        
        if not matrix_files:
            print("No matrix files found!")
            return [], []

        print(f"Found {len(matrix_files)} matrix files to plot")

        # Separate filtered and unfiltered ADMGs
        filtered_files = [f for f in matrix_files if "_all_" not in f.name]
        unfiltered_files = [f for f in matrix_files if "_all_" in f.name]

        print(f"Filtered ADMGs: {len(filtered_files)}")
        print(f"Unfiltered ADMGs: {len(unfiltered_files)}")
        
        return filtered_files, unfiltered_files
    
    def create_output_directory(self, plots_dir: str) -> Path:
        """Create output directory for Python plots."""
        plots_path = Path(plots_dir)
        python_plots_dir = plots_path / "python_plots"
        python_plots_dir.mkdir(exist_ok=True)
        return python_plots_dir


class ADMGProcessor:
    """Main processor for ADMG plotting operations."""
    
    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.plotter = ADMGPlotter(self.config)
        self.file_handler = ADMGFileHandler(self.config)
    
    def plot_all_admgs(self, plots_dir: str = None) -> None:
        """Plot all ADMG matrix files."""
        plots_dir = plots_dir or self.config.default_plots_dir
        
        # Find and categorize files
        filtered_files, unfiltered_files = self.file_handler.find_matrix_files(plots_dir)
        
        if not filtered_files and not unfiltered_files:
            return

        # Create output directory
        python_plots_dir = self.file_handler.create_output_directory(plots_dir)

        # Plot individual ADMGs
        self._plot_individual_admgs(filtered_files, python_plots_dir)
        self._plot_individual_admgs(unfiltered_files, python_plots_dir)

        print(f"\nAll plots saved to: {python_plots_dir}")

        # Create combined plots for each PAG
        print("\nCreating combined plots...")
        self.create_combined_pag_plot("pag1", filtered_files, python_plots_dir)
        self.create_combined_pag_plot("pag2", filtered_files, python_plots_dir)

    def _plot_individual_admgs(self, matrix_files: List[Path], output_dir: Path) -> None:
        """Plot individual ADMG files."""
        for matrix_file in sorted(matrix_files):
            print(f"Processing {matrix_file.name}...")

            # Parse matrix
            matrix, node_names = self.file_handler.parser.parse_matrix_file(matrix_file)

            if matrix is None:
                print(f"Could not parse {matrix_file.name}")
                continue

            # Generate title and output path
            base_name = matrix_file.stem.replace("_matrix", "")
            # Extract just the number from the base name (e.g., "pag1_admg_1" -> "1")
            admg_number = base_name.split("_")[-1]
            title = f"ADMG {admg_number}"
            output_path = output_dir / f"{base_name}_python.png"

            # Plot the ADMG
            self.plotter.plot_admg(matrix, node_names, title, str(output_path))

    def create_combined_pag_plot(self, pag_name: str, matrix_files: List[Path], output_dir: Path) -> None:
        """Create a combined plot showing all ADMGs for a specific PAG."""
        if not matrix_files:
            return

        # Filter files for this specific PAG
        pag_files = [f for f in matrix_files if f.name.startswith(pag_name)]
        if not pag_files:
            return

        # Create combined graph
        combined_graph = self._create_combined_graph(pag_files)
        
        if combined_graph:
            self._save_combined_plot(combined_graph, pag_name, output_dir)

    def _create_combined_graph(self, pag_files: List[Path]) -> Optional[pydot.Dot]:
        """Create the combined graph structure."""
        # Create a large combined graph
        combined_graph = pydot.Dot(graph_type="digraph")
        combined_graph.set_graph_defaults(
            rankdir="TB", 
            size=self.config.combined_graph_size, 
            dpi=str(self.config.image_dpi)
        )

        # Add subgraph for each ADMG
        for matrix_file in sorted(pag_files):
            # Parse matrix
            matrix, node_names = self.file_handler.parser.parse_matrix_file(matrix_file)
            if matrix is None:
                continue

            # Create subgraph for this ADMG
            base_name = matrix_file.stem.replace("_matrix", "")
            admg_number = base_name.split("_")[-1]
            subgraph = self._create_admg_subgraph(matrix, node_names, admg_number)
            combined_graph.add_subgraph(subgraph)

        return combined_graph

    def _create_admg_subgraph(self, matrix: np.ndarray, node_names: List[str], admg_number: str) -> pydot.Subgraph:
        """Create a subgraph for a single ADMG."""
        subgraph_name = f"cluster_{admg_number}"
        subgraph = pydot.Subgraph(graph_name=subgraph_name)
        subgraph.set_label(f"ADMG {admg_number}")

        # Add nodes to subgraph
        for name in node_names:
            node = pydot.Node(
                f"{admg_number}_{name}",
                label=name,
                style="filled",
                fillcolor=self.config.node_color,
                fontname="Arial",
                fontsize=str(self.config.node_font_size_combined),
                fontweight="bold",
            )
            subgraph.add_node(node)

        # Add edges to subgraph
        self._add_subgraph_edges(matrix, node_names, admg_number, subgraph)
        
        return subgraph

    def _add_subgraph_edges(self, matrix: np.ndarray, node_names: List[str], admg_number: str, subgraph: pydot.Subgraph) -> None:
        """Add edges to the subgraph."""
        processed_bidirected = set()
        n = len(node_names)
        
        for row in range(n):
            for col in range(n):
                if matrix[row, col] == 1:  # Directed edge
                    edge = pydot.Edge(
                        f"{admg_number}_{node_names[row]}",
                        f"{admg_number}_{node_names[col]}",
                        color=self.config.edge_color,
                        arrowhead="normal",
                        penwidth=self.config.edge_width_combined,
                    )
                    subgraph.add_edge(edge)
                elif matrix[row, col] == 100:  # Bidirected edge
                    edge_key = tuple(sorted([f"{admg_number}_{node_names[row]}", f"{admg_number}_{node_names[col]}"]))
                    if edge_key not in processed_bidirected:
                        edge = pydot.Edge(
                            f"{admg_number}_{node_names[row]}",
                            f"{admg_number}_{node_names[col]}",
                            color=self.config.edge_color,
                            arrowhead="normal",
                            arrowtail="normal",
                            dir="both",
                            penwidth=self.config.edge_width_combined,
                        )
                        subgraph.add_edge(edge)
                        processed_bidirected.add(edge_key)
                elif matrix[row, col] == 101:  # Treat as directed
                    edge = pydot.Edge(
                        f"{admg_number}_{node_names[row]}",
                        f"{admg_number}_{node_names[col]}",
                        color=self.config.edge_color,
                        arrowhead="normal",
                        penwidth=self.config.edge_width_combined,
                    )
                    subgraph.add_edge(edge)

    def _save_combined_plot(self, combined_graph: pydot.Dot, pag_name: str, output_dir: Path) -> None:
        """Save the combined plot to file."""
        try:
            png_data = combined_graph.create_png()
            img = Image.open(io.BytesIO(png_data))
            
            title = f"All ADMGs for {pag_name.upper()}"
            self.plotter._add_title_to_image(img, title, self.config.title_font_size_combined)
            
            output_path = output_dir / f"{pag_name}_all_admgs.png"
            img.save(output_path, "PNG", dpi=(self.config.image_dpi, self.config.image_dpi))
            
            print(f"Combined plot saved: {output_path}")
            
        except Exception as e:
            print(f"Combined plot failed for {pag_name}: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot ADMGs from matrix files")
    parser.add_argument(
        "--plots-dir", 
        default="plots", 
        help="Directory containing matrix files (default: plots)"
    )
    parser.add_argument(
        "--max-cols", 
        type=int, 
        default=3, 
        help="Maximum columns per row in combined plots (default: 3)"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=300, 
        help="Image DPI (default: 300)"
    )
    parser.add_argument(
        "--font-size", 
        type=int, 
        default=16, 
        help="Node font size (default: 16)"
    )
    return parser.parse_args()


def main():
    """Main function with CLI support."""
    args = parse_arguments()
    
    # Create configuration
    config = PlotConfig(
        default_plots_dir=args.plots_dir,
        max_cols_per_row=args.max_cols,
        image_dpi=args.dpi,
        node_font_size=args.font_size,
    )
    
    # Create processor and run
    processor = ADMGProcessor(config)
    processor.plot_all_admgs()


if __name__ == "__main__":
    print("ADMG Plotting Script")
    print("===================")

    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Run main function
    main()

    print("\nDone!")
