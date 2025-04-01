import os
import pickle
import anndata
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

RESULTS_DIR = "./results"
PROMPTS_PICKLE_PATH = "./prompts.pickle"

def load_samples(results_dir=RESULTS_DIR):
    """
    Load all *_sample.pickle files into memory.
    Returns a dict {sample_filename: AnnData}.
    """
    sample_pickles = sorted(glob(os.path.join(results_dir, "*_sample.pickle")))
    sample_adata_dict = {}
    print("Loading sample pickles:")
    for spkl in sample_pickles:
        print("  ", os.path.basename(spkl))
        with open(spkl, "rb") as f:
            adata_sample = pickle.load(f)
        sample_adata_dict[os.path.basename(spkl)] = adata_sample
    return sample_adata_dict

def load_neighbors_metadata(results_dir=RESULTS_DIR):
    """
    Load all neighbors_metadata_part*.h5ad into a single AnnData.
    We assume they have consistent obs columns.
    """
    meta_files = sorted(glob(os.path.join(results_dir, "neighbors_metadata_part*.h5ad")))
    if not meta_files:
        raise ValueError("No neighbors_metadata_part*.h5ad files found.")
    print("Loading neighbors metadata in parts and concatenating:")
    all_metas = []
    for mf in meta_files:
        print("  ", os.path.basename(mf))
        # read the partial anndata
        partial = anndata.read_h5ad(mf)
        all_metas.append(partial)
    # Concatenate all
    # This may do an outer join on obs/var. If columns differ, you might want join="outer"
    neighbors_full_adata = anndata.concat(all_metas, axis=0, join="outer", merge="unique")
    print("Combined neighbor metadata shape:", neighbors_full_adata.shape)
    return neighbors_full_adata

def load_cell_to_neighbors(results_dir=RESULTS_DIR):
    """
    Load all neighbors_*.pickle and build a mapping
    cell_id -> list of neighbor_ids.

    Return a dictionary: { cell_id: [neighbor_id1, neighbor_id2, ..., neighbor_id30], ... }
    """
    neighbor_files = sorted(glob(os.path.join(results_dir, "neighbors_*.pickle")))
    print("Loading neighbor pickles (cell->neighbors mapping):")
    cell_to_neighbors = {}
    for nf in tqdm(neighbor_files):
        with open(nf, "rb") as f:
            # Expect a DataFrame with columns: ["query_id", "neighbor_ids", ...]
            neighbors_df = pickle.load(f)
        # Build or extend mapping
        for idx, row in neighbors_df.iterrows():
            query_id = row["query_id"]
            neigh_ids = row["neighbor_ids"]  # typically a list of neighbor IDs
            cell_to_neighbors[query_id] = neigh_ids
    return cell_to_neighbors

def build_prompts(
    sample_adata_dict,
    neighbors_full_adata,
    cell_to_neighbors
):
    """
    For each cell in each sample, retrieve the 30 neighbors, build the prompt:
    \"\"\"
    Having following cell:
    [cell metadata]
    and 30 similar cells:
    [neighbors metadata]
    generate 5 sentence textual description of this cell
    \"\"\"
    Return a list of prompt strings.
    """
    prompts = []

    # Convert neighbors_full_adata.obs to DataFrame for easy indexing
    # This allows neighbors_full_adata.obs.loc[id] to fetch the row if id is in .obs_names
    neighbor_meta_df = neighbors_full_adata.obs

    # For each sample:
    for sample_name, adata in sample_adata_dict.items():
        print(f"Building prompts for sample: {sample_name}, shape={adata.shape}")
        sample_df = adata.obs  # metadata for this sample's cells

        # For each cell in the sample
        for cell_id in tqdm(sample_df.index, desc=f"Cells in {sample_name}"):
            # 1) gather the cell's metadata
            #    Convert to a small JSON-like string or a text summary
            cell_metadata_str = sample_df.loc[cell_id].to_dict()

            # 2) find the neighbors
            if cell_id not in cell_to_neighbors:
                # Some chunk might not have found neighbors, skip or continue
                continue

            neighbor_ids = cell_to_neighbors[cell_id]

            # 3) gather neighbor metadata
            #    If any neighbor IDs are missing from neighbors_full_adata, you can skip them
            #    but we expect them to all be in .obs_names
            neighbor_rows = neighbor_meta_df.loc[neighbor_ids, :].to_dict(orient="index")

            # Turn neighbor_rows into a nicely formatted string or partial data
            # We'll just do str(neighbor_rows) for demonstration. You can customize.
            neighbor_metadata_str = str(neighbor_rows)

            # 4) Build the prompt
            prompt_text = f"""Having following cell:
{cell_metadata_str}
and 30 similar cells:
{neighbor_metadata_str}
generate 5 sentence textual description of this cell
"""
            prompts.append(prompt_text)

    return prompts

def main():
    print("1) Loading all samples...")
    sample_adata_dict = load_samples()

    print("\n2) Loading all neighbors metadata...")
    neighbors_full_adata = load_neighbors_metadata()

    print("\n3) Loading all neighbor <-> cell mappings...")
    cell_to_neighbors = load_cell_to_neighbors()

    print("\n4) Building prompts for each cell in each sample...")
    prompts = build_prompts(sample_adata_dict, neighbors_full_adata, cell_to_neighbors)

    print("\n5) Saving prompts to pickle...")
    with open(PROMPTS_PICKLE_PATH, "wb") as f:
        pickle.dump(prompts, f)

    print(f"All done. Saved {len(prompts)} prompts to {PROMPTS_PICKLE_PATH}")

if __name__ == "__main__":
    main()
