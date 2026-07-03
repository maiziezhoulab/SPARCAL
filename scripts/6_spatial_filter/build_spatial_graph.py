import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm

# Configuration dictionaries
SPATIAL_DATA_CONFIGS = {
    "P4_TUMOR": {
        "meta_data_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep{section_id}/Meta_Data",
        "scalefactors_file": "GSM4565823_P4_rep{section_id}_scalefactors_json.json",
        "positions_file": "GSM4565823_P4_rep{section_id}_tissue_positions_list.csv",
        "barcodes_file": "GSM4565823_barcodes.tsv",
        "section_ids": ["1", "2"]
    },
    "P6_TUMOR": {
        "meta_data_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/spaceranger_align_rep{section_id}/Meta_Data",
        "scalefactors_file": "GSM4565825_P6_rep{section_id}_scalefactors_json.json",
        "positions_file": "GSM4565825_P6_rep{section_id}_tissue_positions_list.csv",
        "barcodes_file": "GSM4565825_barcodes.tsv",
        "section_ids": ["1", "2"]
    }
}

FILTERED_BAM_CONFIGS = {
    "P4_TUMOR": {
        "bam_dir": "/data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/{section_id}/output_VCFs/BAM_filtered/{quality_filter}"
    },
    "P6_TUMOR": {
        "bam_dir": "/data/maiziezhou_lab/yuqi/snv_calling/data/P6_tumor/{section_id}/output_VCFs/BAM_filtered/{quality_filter}"
    }
}

@dataclass
class ScalingFactors:
    spot_diameter_fullres: float
    tissue_hires_scalef: float
    fiducial_diameter_fullres: float
    tissue_lowres_scalef: float

@dataclass
class SpotPosition:
    barcode: str
    x: float
    y: float
    is_tissue: bool

@dataclass
class SNVInfo:
    chrom: str
    pos: int
    ref: str
    alt: str
    af: float
    
    def __hash__(self):
        return hash((self.chrom, self.pos, self.ref, self.alt))

class SpatialNode:
    def __init__(self, barcode: str, x: float, y: float):
        self.barcode = barcode
        self.x = x
        self.y = y
        self.snvs: Dict[str, SNVInfo] = {}
        self.neighbors: Set[str] = set()
        
    def add_snv(self, snv_id: str, snv_info: SNVInfo):
        self.snvs[snv_id] = snv_info
        
    def add_neighbor(self, neighbor_barcode: str):
        self.neighbors.add(neighbor_barcode)

class SpatialSNVGraph:
    def __init__(self, scaling_factors: ScalingFactors):
        self.scaling = scaling_factors
        self.nodes: Dict[str, SpatialNode] = {}
        self.grid_index: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self.grid_size = 100  # Grid size for spatial indexing
        
    def add_spot(self, spot: SpotPosition):
        """Add a spatial spot to the graph."""
        if not spot.is_tissue:
            return
            
        node = SpatialNode(spot.barcode, spot.x, spot.y)
        self.nodes[spot.barcode] = node
        
        # Add to spatial grid index
        grid_x = int(spot.x // self.grid_size)
        grid_y = int(spot.y // self.grid_size)
        self.grid_index[(grid_x, grid_y)].add(spot.barcode)
        
    def find_neighbors(self, barcode: str, radius: float) -> Set[str]:
        """Find neighboring spots within given radius."""
        node = self.nodes[barcode]
        neighbors = set()
        
        grid_radius = int(radius // self.grid_size) + 1
        center_x = int(node.x // self.grid_size)
        center_y = int(node.y // self.grid_size)
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                grid_key = (center_x + dx, center_y + dy)
                if grid_key in self.grid_index:
                    for neighbor_barcode in self.grid_index[grid_key]:
                        if neighbor_barcode != barcode:
                            neighbor = self.nodes[neighbor_barcode]
                            dist = np.sqrt(
                                (node.x - neighbor.x)**2 + 
                                (node.y - neighbor.y)**2
                            )
                            if dist <= radius:
                                neighbors.add(neighbor_barcode)
        
        return neighbors
    
    def build_neighbor_graph(self, radius: float):
        """Build neighborhood connections for all spots."""
        for barcode in tqdm(self.nodes.keys(), desc="Building neighbor graph"):
            neighbors = self.find_neighbors(barcode, radius)
            self.nodes[barcode].neighbors = neighbors
            
    def add_snv_to_spot(self, barcode: str, snv_id: str, snv_info: SNVInfo):
        """Add SNV information to a spot."""
        if barcode in self.nodes:
            self.nodes[barcode].add_snv(snv_id, snv_info)
            
    def export_graph_data(self) -> Dict:
        """Export graph data in format suitable for analysis."""
        return {
            'nodes': {
                barcode: {
                    'x': node.x * self.scaling.tissue_hires_scalef,
                    'y': node.y * self.scaling.tissue_hires_scalef,
                    'num_snvs': len(node.snvs),
                    'neighbors': list(node.neighbors)
                }
                for barcode, node in self.nodes.items()
            },
            'scaling_factors': {
                'spot_diameter_fullres': self.scaling.spot_diameter_fullres,
                'tissue_hires_scalef': self.scaling.tissue_hires_scalef,
                'fiducial_diameter_fullres': self.scaling.fiducial_diameter_fullres,
                'tissue_lowres_scalef': self.scaling.tissue_lowres_scalef
            }
        }

class SpatialGraphBuilder:
    def __init__(self, dataset_name: str, section_id: str, quality_filter: str = "baseQ0mapQ0"):
        self.dataset_name = dataset_name
        self.section_id = section_id
        self.quality_filter = quality_filter
        self.validate_config()
        self.setup_paths()
        
    def validate_config(self):
        """Validate dataset configuration."""
        if self.dataset_name not in SPATIAL_DATA_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        config = SPATIAL_DATA_CONFIGS[self.dataset_name]
        if self.section_id not in config["section_ids"]:
            raise ValueError(
                f"Invalid section_id for {self.dataset_name}. "
                f"Valid section IDs are: {config['section_ids']}"
            )
            
    def setup_paths(self):
        """Setup paths for input files."""
        config = SPATIAL_DATA_CONFIGS[self.dataset_name]
        
        # Meta data paths
        self.meta_data_path = config["meta_data_path"].format(section_id=self.section_id)
        self.scalefactors_file = os.path.join(
            self.meta_data_path,
            config["scalefactors_file"].format(section_id=self.section_id)
        )
        self.positions_file = os.path.join(
            self.meta_data_path,
            config["positions_file"].format(section_id=self.section_id)
        )
        self.barcodes_file = os.path.join(
            self.meta_data_path,
            config["barcodes_file"]
        )
        
        # BAM directory
        bam_config = FILTERED_BAM_CONFIGS[self.dataset_name]
        self.bam_dir = bam_config["bam_dir"].format(
            section_id=self.section_id,
            quality_filter=self.quality_filter
        )
        
    def load_scaling_factors(self) -> ScalingFactors:
        """Load scaling factors from JSON file."""
        with open(self.scalefactors_file, 'r') as f:
            data = json.load(f)
            return ScalingFactors(
                spot_diameter_fullres=data["spot_diameter_fullres"],
                tissue_hires_scalef=data["tissue_hires_scalef"],
                fiducial_diameter_fullres=data["fiducial_diameter_fullres"],
                tissue_lowres_scalef=data["tissue_lowres_scalef"]
            )
            
    def load_spot_positions(self) -> List[SpotPosition]:
        """Load spot positions from tissue positions file."""
        positions = []
        df = pd.read_csv(self.positions_file, header=None)
        
        for _, row in df.iterrows():
            positions.append(SpotPosition(
                barcode=row[0],
                x=float(row[1]),
                y=float(row[2]),
                is_tissue=bool(int(row[3]))
            ))
            
        return positions
        
    def build_graph(self, radius: float = 500.0) -> SpatialSNVGraph:
        """Build spatial graph from spot positions."""
        print(f"\nBuilding spatial graph for {self.dataset_name} section {self.section_id}")
        
        # Load scaling factors and positions
        scaling_factors = self.load_scaling_factors()
        spot_positions = self.load_spot_positions()
        
        # Initialize graph
        graph = SpatialSNVGraph(scaling_factors)
        
        # Add spots
        print("Adding spots to graph...")
        for spot in tqdm(spot_positions):
            graph.add_spot(spot).
            '\]]]'
        # Build neighbor connections
        print("Building neighbor connections...")
        graph.build_neighbor_graph(radius)
        
        # Print summary
        print("\nGraph Summary:")
        print(f"Total spots: {len(graph.nodes)}")
        avg_neighbors = np.mean([len(node.neighbors) for node in graph.nodes.values()])
        print(f"Average neighbors per spot: {avg_neighbors:.2f}")
        
        return graph

def main():
    parser = argparse.ArgumentParser(description="Build spatial SNV graph")
    parser.add_argument("--dataset", required=True, choices=["P4_TUMOR", "P6_TUMOR"],
                      help="Dataset to process")
    parser.add_argument("--section-id", required=True,
                      help="Section ID")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                      help="Quality filter to use")
    parser.add_argument("--radius", type=float, default=500.0,
                      help="Radius for neighbor connections")
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = SpatialGraphBuilder(
        dataset_name=args.dataset,
        section_id=args.section_id,
        quality_filter=args.quality_filter
    )
    
    # Build graph
    graph = builder.build_graph(radius=args.radius)
    
    # Export graph data (can be extended based on needs)
    graph_data = graph.export_graph_data()
    
    # Print summary
    print("\nExported Graph Summary:")
    print(f"Number of nodes: {len(graph_data['nodes'])}")
    print(f"Scaling factor: {graph_data['scaling_factors']['tissue_hires_scalef']}")

if __name__ == "__main__":
    main()

# Usage examples:
# For P4_TUMOR:
# python scripts/graph_build/run_spatial_graph.py --dataset P4_TUMOR --section-id 1 --quality-filter baseQ0mapQ0
# 
# For P6_TUMOR:
# python scripts/graph_build/run_spatial_graph.py --dataset P6_TUMOR --section-id 1 --quality-filter baseQ0mapQ0