import os
import random
import pickle
import wandb
from tqdm import tqdm

import anndata
import numpy as np
import scanpy as sc
import scvi

import cellxgene_census
import cellxgene_census.experimental

###############################
# User-defined global settings
###############################

DATA_DIR = "./data"            # location of .h5ad files (non-recursive)
MODEL_DIR = "./model"         # scVI model directory
OUTPUT_DIR = "./results"      # where to put results
CENSUS_VERSION = "2024-07-01" # which cellxgene census to use
SAMPLE_SIZE_PER_FILE = 20    # how many cells to sample from each .h5ad
BATCH_SIZE = 10               # how many cells per partial batch
METADATA_BATCH_SIZE = 1_000   # how many neighbor IDs to fetch at a time
MAX_FILES = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

##########################
# 1. Collect input files
##########################

def list_h5ad_files(data_dir=DATA_DIR):
    """List .h5ad files (non-recursive) in data_dir."""
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".h5ad") and os.path.isfile(os.path.join(data_dir, f))
    ]
    return sorted(files)


###############################
# 2. Sampling from each file
###############################

def sample_data_from_file(fpath, sample_size=SAMPLE_SIZE_PER_FILE):
    """
    Read a single .h5ad file into memory,
    randomly sample 'sample_size' cells, and return the new AnnData.
    """
    adata = sc.read_h5ad(fpath)
    if sample_size > adata.n_obs:
        raise ValueError(
            f"Requested sample_size={sample_size} but the file {fpath} has only {adata.n_obs} cells."
        )
    idx = np.random.choice(adata.n_obs, size=sample_size, replace=False)
    sampled = adata[idx, :].copy()
    # Let’s store the original cell IDs to keep track:
    # We use .obs_names plus the dataset name for a global identifier:
    dataset_id = os.path.basename(fpath)
    # Create a column with the original row index from the file
    sampled.obs["original_cell_id"] = sampled.obs_names
    sampled.obs["source_dataset"] = dataset_id
    # Overwrite .obs_names with something unique per file/cell
    # E.g. "dataset##original_cell_id"
    new_obs_names = [
        f"{dataset_id}##{x}" for x in sampled.obs["original_cell_id"].values
    ]
    sampled.obs_names = new_obs_names
    return sampled


###############################
# 3. Prepare scVI query model
###############################

def run_scvi_query(adata, model_dir=MODEL_DIR):
    """
    - Prepares an AnnData batch for scVI
    - Embeds into latent space
    - Returns the updated AnnData with .obsm["scvi"]
    """
    # scVI typically needs to match the same var features that
    # the model was trained on. Check for duplicates, etc.
    adata.var_names_make_unique()

    # Minimal columns that scVI expects:
    if "n_counts" not in adata.obs:
        adata.obs["n_counts"] = adata.X.sum(axis=1)
    if "batch" not in adata.obs:
        adata.obs["batch"] = "unassigned"

    # Prepare query
    scvi.model.SCVI.prepare_query_anndata(adata, model_dir)
    vae_q = scvi.model.SCVI.load_query_data(adata, model_dir)
    vae_q.is_trained = True

    # Latent representation
    latent = vae_q.get_latent_representation()
    adata.obsm["scvi"] = latent

    return adata


###############################################
# 4. Find neighbors for a small batch of cells
###############################################

def find_neighbors_for_batch(adata_batch, census_version=CENSUS_VERSION):
    """
    Use cellxgene_census.experimental.find_nearest_obs
    on the provided batch (adata_batch).
    Return the neighbors DataFrame.
    """
    # Must ensure "feature_name" is used as .var.index for Census
    if "feature_name" in adata_batch.var.columns:
        # filter out missing
        keep_mask = adata_batch.var["feature_name"].notnull()
        adata_batch = adata_batch[:, keep_mask].copy()
        adata_batch.var.set_index("feature_name", inplace=True)
    neighbors = cellxgene_census.experimental.find_nearest_obs(
        embedding_name="scvi",
        organism="mus_musculus",
        census_version=census_version,
        query=adata_batch,
        k=30,
        memory_GiB=16,
        nprobe=20,
    )
    return neighbors


##############################
# 5. Write partial neighbors
##############################

def store_neighbors(neighbors, batch_name, out_dir=OUTPUT_DIR):
    """
    Store the neighbor result in a pickle so we can retrieve later.
    """
    out_pickle = os.path.join(out_dir, f"neighbors_{batch_name}.pickle")
    with open(out_pickle, "wb") as f:
        pickle.dump(neighbors, f)


################################
# 6. Pass 1: neighbor finding
################################

def process_datasets_for_neighbors(
    h5ad_files,
    batch_size=BATCH_SIZE,
    sample_size=SAMPLE_SIZE_PER_FILE,
    model_dir=MODEL_DIR
):
    """
    For each file:
      - sample
      - break into BATCH_SIZE chunks
      - scVI embed each chunk
      - find neighbors
      - store partial results
    """
    pbar_datasets = tqdm(h5ad_files, desc="Datasets", position=0)
    if len(pbar_datasets) > MAX_FILES:
        pbar_datasets = pbar_datasets[:MAX_FILES]
    for fpath in pbar_datasets:
        pbar_datasets.set_postfix_str(os.path.basename(fpath))

        # Load & sample data from this file
        adata_sample = sample_data_from_file(fpath, sample_size)

        # Break the sample into BATCH_SIZE chunks
        obs_names = adata_sample.obs_names.tolist()
        total_cells = len(obs_names)

        # Simple chunking
        for start_idx in range(0, total_cells, batch_size):
            end_idx = start_idx + batch_size
            # we handle partial batch if end_idx > total_cells
            chunk_obs_names = obs_names[start_idx:end_idx]
            adata_chunk = adata_sample[chunk_obs_names, :].copy()

            # scVI embedding
            adata_chunk = run_scvi_query(adata_chunk, model_dir=model_dir)

            # find neighbors
            neighbors = find_neighbors_for_batch(adata_chunk)

            # store partial neighbor results
            batch_name = f"{os.path.basename(fpath)}_{start_idx}-{end_idx}"
            store_neighbors(neighbors, batch_name)

            # Log progress to wandb
            wandb.log({"neighbors_batches_done": 1})

            # Optional: Free memory
            del adata_chunk


########################################
# 7. Gather neighbor IDs after pass 1
########################################

def gather_all_neighbor_ids(neighbor_pickles_dir=OUTPUT_DIR):
    """
    Read every 'neighbors_*.pickle' file in neighbor_pickles_dir,
    accumulate all neighbor IDs into a global set,
    also keep track of cell->neighbors (for final references).
    Return:
      - neighbor_ids (set of all neighbors)
      - cell_to_neighbors (dict of "cell_name" -> list of neighbor_ids)
    """
    neighbor_ids_set = set()
    cell_to_neighbors = {}

    files = [
        f for f in os.listdir(neighbor_pickles_dir)
        if f.startswith("neighbors_") and f.endswith(".pickle")
    ]
    for fname in tqdm(files, desc="Gather neighbor IDs"):
        full_path = os.path.join(neighbor_pickles_dir, fname)
        with open(full_path, "rb") as f:
            neighbors_df = pickle.load(f)
        # neighbors_df has columns like:
        #   query_id, neighbor_rank, neighbor_id, distance, ...
        # We want to gather neighbor_id in a set
        for row in neighbors_df.itertuples():
            neighbor_ids_set.add(row.neighbor_id)
            # Also store the mapping from cell->neighbors
            # row.query_id is the cell name
            if row.query_id not in cell_to_neighbors:
                cell_to_neighbors[row.query_id] = []
            cell_to_neighbors[row.query_id].append(row.neighbor_id)

    return neighbor_ids_set, cell_to_neighbors


##################################################
# 8. Fetch neighbor metadata in batches (Pass 2)
##################################################

def fetch_neighbors_metadata_in_batches(
    neighbor_ids_set,
    cell_to_neighbors,
    census_version=CENSUS_VERSION,
    batch_size=METADATA_BATCH_SIZE,
    out_dir=OUTPUT_DIR
):
    """
    Take the set of neighbor IDs and fetch metadata from the Census in chunks.
    Store partial anndata files so we don't lose progress if there's an error.

    Also store a "neighbor_id -> file_part" map so we know which file
    the neighbor metadata is in, if needed.
    """
    neighbor_ids_list = sorted(list(neighbor_ids_set))
    total_neighbors = len(neighbor_ids_list)
    print(f"Total unique neighbor IDs = {total_neighbors}")

    neighbor_id_to_filepart = {}
    with cellxgene_census.open_soma(census_version=census_version) as census:
        pbar = tqdm(range(0, total_neighbors, batch_size), desc="Fetch metadata")
        part_index = 0
        for start_idx in pbar:
            end_idx = start_idx + batch_size
            chunk_ids = neighbor_ids_list[start_idx:end_idx]

            # Download chunk
            neighbors_meta_adata = cellxgene_census.get_anndata(
                census,
                organism="mus_musculus",
                measurement_name="RNA",
                obs_coords=chunk_ids,
                var_column_names=None
            )

            # Save partial
            part_fname = os.path.join(out_dir, f"neighbors_metadata_part{part_index}.h5ad")
            neighbors_meta_adata.write_h5ad(part_fname)

            # Map each neighbor_id to this part file
            for nid in chunk_ids:
                neighbor_id_to_filepart[nid] = part_fname

            part_index += 1

            # log progress to wandb
            wandb.log({"metadata_downloaded": len(chunk_ids)})

    # Also store the dictionary "neighbor_id -> filepart" so we can find them
    map_pickle = os.path.join(out_dir, "neighbor_id_to_filepart_map.pickle")
    with open(map_pickle, "wb") as f:
        pickle.dump(neighbor_id_to_filepart, f)

    # Finally, store the entire cell->neighbors mapping in a stable location
    c2n_pickle = os.path.join(out_dir, "cell_to_neighbors_map.pickle")
    with open(c2n_pickle, "wb") as f:
        pickle.dump(cell_to_neighbors, f)


##############################
# 9. Main orchestrating logic
##############################

def main():
    # Initialize wandb
    wandb.init(project="my_project", name="multi_file_annotator")

    # 1) Collect .h5ad files
    h5ad_files = list_h5ad_files(DATA_DIR)
    print("Found files:")
    for f in h5ad_files:
        print("  ", f)

    # 2) Pass 1: For each dataset, sample, chunk in BATCH_SIZE,
    #    find neighbors, store partial results
    process_datasets_for_neighbors(h5ad_files)

    # 3) After pass 1, gather all neighbor IDs
    neighbor_ids_set, cell_to_neighbors = gather_all_neighbor_ids()

    # 4) Pass 2: In batches, fetch neighbor metadata from Census
    fetch_neighbors_metadata_in_batches(neighbor_ids_set, cell_to_neighbors)

    # Done
    print("All done. Intermediate files are in:", OUTPUT_DIR)
    wandb.finish()


if __name__ == "__main__":
    main()
