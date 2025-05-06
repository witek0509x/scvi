import os
import re
import pickle
import anndata
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

RESULTS_DIR = "./results_test"
PROMPTS_PICKLE_PATH = "./prompts_raw.pickle"

def load_all_samples(results_dir=RESULTS_DIR):
    """
    Returns a dict:
      { dataset_prefix_without_dot_h5ad : AnnData }
    where the dataset_prefix is derived from the file name
    'XYZ_sample.pickle' -> prefix 'XYZ'.
    """
    sample_files = sorted(glob(os.path.join(results_dir, "*_sample.pickle")))
    sample_adata_dict = {}

    # Regex to capture <prefix>_sample.pickle
    # e.g. "8c64b76f-...-a4c69be77325_sample.pickle" -> prefix "8c64b76f-...-a4c69be77325"
    sample_pat = re.compile(r"^(.*)_sample\.pickle$")

    print("Loading sample pickles:")
    for sf in sample_files:
        fname = os.path.basename(sf)
        match = sample_pat.match(fname)
        if not match:
            continue
        prefix = match.group(1)  # e.g. "8c64b76f-...."
        print("  ", fname, "-> prefix:", prefix)
        with open(sf, "rb") as f:
            adata_sample = pickle.load(f)
        sample_adata_dict[prefix] = adata_sample

    return sample_adata_dict

def load_neighbors_metadata(results_dir=RESULTS_DIR):
    """
    Concatenate all neighbors_metadata_partX.h5ad files into a single AnnData.
    """
    meta_files = sorted(glob(os.path.join(results_dir, "neighbors_metadata_part*.h5ad")))
    if not meta_files:
        raise ValueError("No neighbors_metadata_part*.h5ad files found. Check your directory.")
    print("Loading neighbors metadata (all parts) and concatenating:")
    all_parts = []
    for mf in meta_files:
        print("  ", os.path.basename(mf))
        part_adata = anndata.read_h5ad(mf)
        all_parts.append(part_adata)
    neighbors_full_adata = anndata.concat(all_parts, axis=0, join="outer", merge="unique")
    print("Combined neighbor metadata shape:", neighbors_full_adata.shape)
    return neighbors_full_adata

def build_cell_to_neighbors_mapping(sample_adata_dict, results_dir=RESULTS_DIR):
    """
    For each dataset prefix in sample_adata_dict, look for neighbor chunk pickles
    named neighbors_<prefix>.h5ad_[start]-[stop].pickle

    Then reconstruct a dictionary: cell_id -> neighbor_ids.

    Returns a dict { cell_id : [neighborID1, neighborID2, ... neighborID30], ... }.
    """
    # The pattern for neighbor chunk files is:
    # neighbors_<prefix>.h5ad_[start]-[stop].pickle
    # We'll parse <prefix> plus start/stop, so we know which rows these neighbor IDs correspond to
    # in that sample's obs_names (which we chunked in the original script).

    # Regex to capture: neighbors_<prefix>.h5ad_<start>-<stop>.pickle
    neigh_pat = re.compile(r"^neighbors_(.*)\.h5ad_(\d+)-(\d+)\.pickle$")

    # We'll accumulate all neighbor chunk files in a list
    neighbor_chunk_files = sorted(glob(os.path.join(results_dir, "neighbors_*.pickle")))

    # We'll keep a single global dict { cell_id -> [neighbor_ids...] }
    # even though each prefix has a separate set of cells
    cell_to_neighbors = {}

    print("Loading neighbor chunk pickles to build cell->neighbors mapping:")
    for chunk_file in tqdm(neighbor_chunk_files):
        fname = os.path.basename(chunk_file)
        m = neigh_pat.match(fname)
        if not m:
            # Possibly skip or keep scanning
            continue
        file_prefix_with_h5ad = m.group(1)   # e.g. "8c64b76f-...-a4c69be77325"
        start_idx = int(m.group(2))
        end_idx   = int(m.group(3))

        # The original script's prefix was "8c64b76f-...-a4c69be77325" (without ".h5ad").
        # However, the chunk file has "neighbors_<prefix>.h5ad_..."
        # So we remove the trailing ".h5ad" from file_prefix_with_h5ad:
        if file_prefix_with_h5ad.endswith(".h5ad"):
            dataset_prefix = file_prefix_with_h5ad.replace(".h5ad", "")
        else:
            dataset_prefix = file_prefix_with_h5ad

        # Now let's get the sample anndata for that prefix:
        if dataset_prefix not in sample_adata_dict:
            # Could happen if there's mismatch in names. We'll skip if not found.
            continue
        adata_sample = sample_adata_dict[dataset_prefix]
        # The chunk of cells that was processed in [start_idx : end_idx]:
        sample_obs_names = adata_sample.obs_names.tolist()
        chunk_obs_names = sample_obs_names[start_idx : end_idx]
        # The number of cells in this chunk:
        chunk_size = len(chunk_obs_names)
        if chunk_size == 0:
            # No cells for that range? skip
            continue

        # Load neighbor object from the chunk file
        with open(os.path.join(results_dir, fname), "rb") as f:
            # Suppose we get a DataFrame or some named structure
            # that has an attribute or column "neighbor_ids"
            # which is a 2D numpy array: shape (chunk_size, 30)
            neighbors_obj = pickle.load(f)
            # If it's a DataFrame: neighbors_obj["neighbor_ids"]
            # or if it's an attribute: neighbors_obj.neighbor_ids
            # We must adapt to your actual object structure.

            # For demonstration, let's assume 'neighbors_obj' is a DataFrame
            # and the array is in neighbors_obj["neighbor_ids"].
            # If you have something else, adapt accordingly.
            if isinstance(neighbors_obj, pd.DataFrame):
                # Possibly neighbors_obj['neighbor_ids'] is a 2D array
                neighbor_ids_array = neighbors_obj["neighbor_ids"].values
            else:
                # Another possible structure:
                # neighbors_obj might be a dict with key "neighbor_ids",
                # or a custom class with .neighbor_ids
                # For example:
                neighbor_ids_array = neighbors_obj.neighbor_ids  # adapt as needed

        if neighbor_ids_array.shape[0] != chunk_size:
            # This might be an error if they do not match
            raise ValueError(
                f"Mismatch between chunk size {chunk_size} and neighbors array shape {neighbor_ids_array.shape}"
            )

        # Each row i in neighbor_ids_array corresponds to chunk_obs_names[i]
        # neighbor_ids_array[i] => array of 30 neighbor IDs for that cell
        for i in range(chunk_size):
            cell_id = chunk_obs_names[i]
            cell_neighbors = neighbor_ids_array[i]
            # store in global dict
            cell_to_neighbors[cell_id] = cell_neighbors

    return cell_to_neighbors

def build_prompts(
    sample_adata_dict,
    neighbors_full_adata,
    cell_to_neighbors
):
    """
    For each cell in each sample, we retrieve that cell's 30 neighbors, then
    build the final prompt.

    returns a list of strings (the prompts).
    """
    # We'll just gather the obs DataFrame for neighbors_full_adata
    neighbor_obs_df = neighbors_full_adata.obs
    neighbor_obs_df.set_index("soma_joinid", inplace=True)

    all_prompts = []

    # For each dataset prefix:
    for prefix, adata_sample in sample_adata_dict.items():
        print(f"Building prompts for prefix={prefix}, sample shape={adata_sample.shape}")
        sample_obs_df = adata_sample.obs


        # For each cell in the sample:
        for cell_id in tqdm(sample_obs_df.index, desc=f"Cells in {prefix}"):
            # If no neighbor data for this cell, skip
            if cell_id not in cell_to_neighbors:
                continue
            # 1) Cell's own metadata
            cell_meta_str = sample_obs_df.loc[cell_id].to_dict()

            # 2) Get neighbors
            neighbor_ids = cell_to_neighbors[cell_id]
            # neighbor_ids should be length ~30
            # fetch their metadata from neighbors_full_adata.obs
            # some IDs may not exist in neighbor_obs_df; we can filter them:
            valid_neighbor_ids = [nid for nid in neighbor_ids if nid in neighbor_obs_df.index]
            neighbors_dict = neighbor_obs_df.loc[valid_neighbor_ids].to_dict(orient="index")
            if len(valid_neighbor_ids) == 0:
                print(f"No neighbors for cell {cell_id}, {neighbor_ids}")

            # Build your text prompt:
            prompt = f"""
You have metadata about a single cell.
Your task is to produce a concise, descriptive summary that:

1. Identifies the cell type, tissue of origin, assay used and any significant permutation that was applied to the organism
2. Concludes with the likely role or function of the cell based on the 
   provided data (e.g., regenerative capacity, specialized function, etc.).

### Background Data
- **Target Cell Data**: 
{cell_meta_str}
- Write **1–3 sentences** in a descriptive style, not just bullet points.
- Incorporate **at least one** numeric value (e.g., total UMI counts, number of genes).
- Vary your word choice if possible (use synonyms or alternate phrasing), while staying factually correct.
- Do **not** add information that isn’t supported by the data.

Now, use the information provided above to create a similarly concise yet descriptive passage.
"""
            all_prompts.append(prompt)

    return all_prompts

def main():
    print("1) Loading all sample pickles...")
    sample_adata_dict = load_all_samples()

    print("\n2) Loading full neighbors metadata (concatenate all parts)...")
    neighbors_full_adata = load_neighbors_metadata()

    print("\n3) Building cell->neighbors mapping from chunked neighbor pickles...")
    cell_to_neighbors = build_cell_to_neighbors_mapping(sample_adata_dict)

    print("\n4) Constructing prompts for each cell in each sample...")
    prompts = build_prompts(sample_adata_dict, neighbors_full_adata, cell_to_neighbors)
    print(f"   Built {len(prompts)} total prompts.")

    print("\n5) Saving prompts to:", PROMPTS_PICKLE_PATH)
    with open(PROMPTS_PICKLE_PATH, "wb") as f:
        pickle.dump(prompts, f)

    print("All done.")

if __name__ == "__main__":
    main()
