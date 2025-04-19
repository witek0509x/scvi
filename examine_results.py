#!/usr/bin/env python
import os
import pickle
import sys
from pprint import pprint

import scanpy as sc
import anndata
import numpy as np
import pandas as pd

# Set the directory where the results are stored
RESULTS_DIR = "./results_full"

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def load_and_examine_sample_pickle(filepath, n_samples=5):
    """Load a sample pickle file and print first few cells."""
    print(f"Loading: {filepath}")
    try:
        with open(filepath, "rb") as f:
            adata = pickle.load(f)
        
        print(f"AnnData object: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print("\nObservation names (first {n_samples}):")
        print(adata.obs_names[:n_samples].tolist())
        
        print("\nAvailable observation metadata:")
        print(list(adata.obs.columns))
        
        if len(adata.obs) > 0 and n_samples > 0:
            print(f"\nSample of metadata (first {min(n_samples, len(adata.obs))} cells):")
            print(adata.obs.head(n_samples))
        
        return adata
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def load_and_examine_neighbors_pickle(filepath, n_samples=5):
    """Load a neighbors pickle file and print first few neighbors."""
    print(f"Loading: {filepath}")
    try:
        with open(filepath, "rb") as f:
            neighbors = pickle.load(f)
        
        print(f"Neighbors object type: {type(neighbors)}")
        
        # Check if it's a DataFrame (as in the script's structure)
        if hasattr(neighbors, "shape"):
            print(f"Shape: {neighbors.shape}")
            if hasattr(neighbors, "columns"):
                print(f"Columns: {list(neighbors.columns)}")
            
            if hasattr(neighbors, "head"):
                print("\nFirst few rows:")
                print(neighbors.head(n_samples))
        
        # If it has neighbor_ids attribute (as used in the script)
        if hasattr(neighbors, "neighbor_ids"):
            print("\nSample of neighbor IDs (first few cells, first few neighbors):")
            for i, row in enumerate(neighbors.neighbor_ids[:n_samples]):
                print(f"Cell {i}: {row[:5]} ...")
        
        return neighbors
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def load_and_examine_h5ad(filepath, n_samples=5):
    """Load an h5ad file and print summary information."""
    print(f"Loading: {filepath}")
    try:
        adata = sc.read_h5ad(filepath)
        
        print(f"AnnData object: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        print(f"\nObservation names (first {n_samples}):")
        print(adata.obs_names[:n_samples].tolist())
        
        print("\nAvailable observation metadata:")
        print(list(adata.obs.columns))
        
        if len(adata.obs) > 0 and n_samples > 0:
            print(f"\nSample of metadata (first {min(n_samples, len(adata.obs))} cells):")
            print(adata.obs.head(n_samples))
        
        return adata
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def load_and_examine_mapping_pickle(filepath, n_samples=10):
    """Load the neighbor ID to filepart mapping and print a sample."""
    print(f"Loading: {filepath}")
    try:
        with open(filepath, "rb") as f:
            mapping = pickle.load(f)
        
        print(f"Mapping contains {len(mapping)} entries")
        
        print("\nSample entries:")
        sample_items = list(mapping.items())[:n_samples]
        for neighbor_id, filepart in sample_items:
            print(f"{neighbor_id} -> {os.path.basename(filepart)}")
        
        return mapping
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def main():
    """Main function to examine all result files."""
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory '{RESULTS_DIR}' does not exist.")
        sys.exit(1)
    
    # List all files in the results directory
    all_files = os.listdir(RESULTS_DIR)
    print(f"Found {len(all_files)} files in {RESULTS_DIR}")
    
    # Group files by type
    sample_pickles = [f for f in all_files if f.endswith('_sample.pickle')]
    neighbors_pickles = [f for f in all_files if f.startswith('neighbors_') and f.endswith('.pickle') 
                        and not f == 'neighbor_id_to_filepart_map.pickle']
    metadata_h5ads = [f for f in all_files if f.startswith('neighbors_metadata_') and f.endswith('.h5ad')]
    mapping_file = 'neighbor_id_to_filepart_map.pickle' if 'neighbor_id_to_filepart_map.pickle' in all_files else None
    
    # Print summary of files found
    print(f"Sample pickles: {len(sample_pickles)}")
    print(f"Neighbors pickles: {len(neighbors_pickles)}")
    print(f"Metadata H5AD files: {len(metadata_h5ads)}")
    print(f"Mapping file found: {mapping_file is not None}")
    
    # Examine sample pickles
    if sample_pickles:
        print_separator("SAMPLE PICKLES")
        sample_file = os.path.join(RESULTS_DIR, sample_pickles[0])
        load_and_examine_sample_pickle(sample_file)
    
    # Examine neighbors pickles
    if neighbors_pickles:
        print_separator("NEIGHBORS PICKLES")
        neighbor_file = os.path.join(RESULTS_DIR, neighbors_pickles[0])
        load_and_examine_neighbors_pickle(neighbor_file)
    
    # Examine metadata H5AD files
    if metadata_h5ads:
        print_separator("METADATA H5AD FILES")
        metadata_file = os.path.join(RESULTS_DIR, metadata_h5ads[0])
        load_and_examine_h5ad(metadata_file)
    
    # Examine mapping file
    if mapping_file:
        print_separator("NEIGHBOR ID TO FILEPART MAPPING")
        mapping_path = os.path.join(RESULTS_DIR, mapping_file)
        load_and_examine_mapping_pickle(mapping_path)

if __name__ == "__main__":
    main() 