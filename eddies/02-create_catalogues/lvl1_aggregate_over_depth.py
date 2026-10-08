# %%
import argparse

import pandas as pd
import numpy as np
import os
import glob

# %%
import dask
from dask.distributed import Client, LocalCluster

# %%

parser = argparse.ArgumentParser()
parser.add_argument("lake_name", type=str)
args = parser.parse_args()
lake = args.lake_name

folder_path = rf"/storage/alplakes_test/{lake}_100m_2025/outputs_swirl"
input_folder = os.path.join(folder_path, "eddy_catalogues_final")

output_folder = os.path.join(folder_path, "eddy_catalogues_final")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "lvl1.csv")

# %%
lvl0_csv_path = os.path.join(input_folder, "lvl0.csv")
lake_csv_path = os.path.join(input_folder, "lake_characteristics.csv")

# %%
nb_cores = 30
cluster = LocalCluster(
    n_workers=nb_cores,
    threads_per_worker=1,
    processes=True,
)
client = Client(cluster)

# %% [markdown]
# # Import catalogue lvl0

# %%
df_lvl0 = pd.read_csv(lvl0_csv_path)

# %%
df_lvl0 = df_lvl0.set_index('id', drop=False)

# %% [markdown]
# # Create catalogue lvl1

# %%
# Function to identify the different depths of an eddy
def identify_eddy(level0_data, idx_already_aggregated, id_eddy, dist_threshold, depth_threshold):
    mask = (
            (~level0_data['id'].isin(idx_already_aggregated)) &
            (level0_data['date'] == level0_data.at[id_eddy, 'date']) &
            (level0_data['rotation_direction'] == level0_data.at[id_eddy, 'rotation_direction']) &
            (np.sqrt((level0_data['xc'] - level0_data.at[id_eddy, 'xc'])**2 + (level0_data['yc'] - level0_data.at[id_eddy, 'yc'])**2) < dist_threshold)
        )

    filtered_eddies = level0_data.loc[mask]
    sorted_by_depth = filtered_eddies.sort_values('depth_[m]', ascending=False).reset_index(drop=True)

    for i in range(1,len(sorted_by_depth)):
        if sorted_by_depth.iloc[i]['depth_index'] - sorted_by_depth.iloc[i-1]['depth_index']> depth_threshold:
            drop_idxs = sorted_by_depth.index[i:]
            sorted_by_depth = sorted_by_depth.drop(index=drop_idxs)
            break

    return sorted_by_depth

# %%
@dask.delayed
def aggregate_over_depth(df_lvl0):
    # Parameters
    dist_threshold = 5  # in number of cells
    depth_threshold = 5 # in number of cells
    eddy_rows_lvl1 = []  # Collect aggregated rows here
    id_level1 = 0
    idx_already_aggregated = set()

    # Main loop
    for idx in df_lvl0['id']:
        if idx in idx_already_aggregated:
            continue

        aggregated_data = identify_eddy(df_lvl0, idx_already_aggregated, idx, dist_threshold, depth_threshold)

        row = {
            'id': id_level1,
            'id_lvl0': aggregated_data['id'].tolist(),
            'time_index': aggregated_data.at[0, 'time_index'],
            'date': aggregated_data.at[0, 'date'],
            'depth_min_[m]': aggregated_data['depth_[m]'].min(),
            'depth_max_[m]': aggregated_data['depth_[m]'].max(),
            'xc_mean': aggregated_data['xc'].mean(),
            'yc_mean': aggregated_data['yc'].mean(),
            'surface_area_mean_[m2]': aggregated_data['surface_area_[m2]'].mean(),
            'volume_[m3]': aggregated_data['volume_slice_[m3]'].sum(),
            'rotation_direction': aggregated_data.at[0, 'rotation_direction'],
            'kinetic_energy_eddy_[MJ]': aggregated_data['kinetic_energy_eddy_[MJ]'].sum()
        }

        eddy_rows_lvl1.append(row)
        idx_already_aggregated.update(aggregated_data['id'].tolist())
        id_level1 += 1

    return eddy_rows_lvl1

# %%
task_dict = {}
for idx_time in np.unique(df_lvl0['time_index']):
    task_dict[idx_time] = aggregate_over_depth(df_lvl0[df_lvl0['time_index'] == idx_time])

# %%
results = dask.compute(*task_dict.values())

# %%
# Flatten the list of lists
flattened_rows = [row for sublist in results for row in sublist]

# Create the final DataFrame
df_catalogue_level1 = pd.DataFrame(flattened_rows)

# %%
df_catalogue_level1['id'] = np.arange(len(df_catalogue_level1))

# %% [markdown]
# # Save 

# %%
df_catalogue_level1.to_csv(output_path, index=False)

# %% [markdown]
# # Shut down cluster

# %%
client.close()
cluster.close()


