import numpy as np
import torch
from torch.utils.data import random_split
import os
from datetime import datetime, timedelta

# Define the base directory paths
base_dir = "/lustre/home/sasha/GPM/MATCH.GMI.GPM.V05"

# Function to process brightness temperature data and create features
def process_tb(tb_array):
    center_pixel_index = 9  # The center pixel in 103-117 arrays is 9
    tb_center = tb_array[center_pixel_index]  # First 13 features: TB of the center pixel in 13 channels
    
    # Next 78 features: TB differences for each of the 78 channel pairs
    tb_diffs = np.array([tb_array[center_pixel_index, i] - tb_array[center_pixel_index, j] 
                         for i in range(12) for j in range(i+1, 13)]).flatten()

    # Last 13 features: Standard deviation of TB for the 8 surrounding pixels
    surrounding_indices = list(range(center_pixel_index - 4, center_pixel_index)) + list(range(center_pixel_index + 1, center_pixel_index + 5))
    tb_stds = np.array([np.std(tb_array[surrounding_indices, i]) for i in range(13)])

    # Combine all features into a single array
    features = np.concatenate((tb_center, tb_diffs, tb_stds))
    return features



# Function to load and filter data for one .npy file
def filter_data(tb1_file, tb2_file, vfracConv_file, surfPrecip_file, surfTypeIndex_file):
    # Load the data
    tb1_data = np.load(tb1_file)
    tb2_data = np.load(tb2_file)
    convective_fraction_data = np.load(vfracConv_file)
    rain_rate_data = np.load(surfPrecip_file)
    surface_type_data = np.load(surfTypeIndex_file)
    
    # Assume center pixel index is 29 for the cropped 083-137 arrays
    center_pixel_index = 29

    # Assume center pixel index is 9 for the cropped 103-117 arrays
    center_pixel_index2 = 9
    
    # Find indices where the center pixel rain rate is not 0.0
    valid_precip_indices = rain_rate_data[:, center_pixel_index] > 0.1
    valid_surface_type_indices = surface_type_data[:, center_pixel_index2] == 1

    # Find indices where both the center pixel rain rate is not 0.0 and the center pixel is over ocean
    valid_indices = np.logical_and(valid_precip_indices, valid_surface_type_indices)
    
    if convective_fraction_data.size == 0 or \
       not all(data.any() for data in [tb1_data, tb2_data, convective_fraction_data, rain_rate_data, surface_type_data]):
        print("Warning: Some data arrays are empty or contain all zeros.")
        return None, None  # Return None for both rain rate and rain type datasets

    # Filter out data with no rain at the center pixel
    tb1_data = tb1_data[valid_indices]
    tb2_data = tb2_data[valid_indices]
    convective_fraction_data = convective_fraction_data[valid_indices]
    rain_rate_data = rain_rate_data[valid_indices]
    surface_type_data = surface_type_data[valid_indices]
    
    # Combine the first 9 channels of Tc and the last 4 channels of TcS2
    tb_combined = np.concatenate((tb1_data, tb2_data), axis=-1)

    # Preprocess the brightness temperature data to create features
    tb_features = np.array([process_tb(tb) for tb in tb_combined])
    
    # Binary classification labels for rain type (convective or stratiform)
    rain_type_labels = (convective_fraction_data[:, center_pixel_index] >= 0.5).astype(int)
    rain_type_labels = rain_type_labels.reshape(-1, 1)  # Reshape to ensure it's a 2D array

    # Rain rate labels
    rain_rate_labels = rain_rate_data[:, center_pixel_index].reshape(-1, 1)
    # Balance the dataset
    stratiform_indices = np.where(rain_type_labels == 0)[0]
    convective_indices = np.where(rain_type_labels == 1)[0]

    # Calculate the number of stratiform samples to keep (55% of the original)
    num_stratiform_to_keep = int(len(stratiform_indices) * 0.55)

    # Randomly select stratiform indices to keep
    np.random.seed(42)  # For reproducibility
    stratiform_indices_to_keep = np.random.choice(stratiform_indices, num_stratiform_to_keep, replace=False)

    # Combine the selected stratiform indices with all convective indices
    selected_indices = np.concatenate((stratiform_indices_to_keep, convective_indices))

    # Filter all data based on the selected indices
    tb_features = tb_features[selected_indices]
    rain_type_labels = rain_type_labels[selected_indices]
    rain_rate_labels = rain_rate_labels[selected_indices]

    # Creating separate datasets
    rain_rate_dataset = (tb_features, rain_rate_labels)
    rain_type_dataset = (tb_features, rain_type_labels)

    # Check for consistency in data size before returning
    if tb_features.shape[0] == rain_rate_labels.shape[0] and tb_features.shape[0] == rain_type_labels.shape[0]:
        return rain_rate_dataset, rain_type_dataset
    else:
        print("Warning: Data arrays do not match in size.")
        print(f"TB features shape: {tb_features.shape}")
        print(f"Rain rate labels shape: {rain_rate_labels.shape}")
        print(f"Rain type labels shape: {rain_type_labels.shape}")
        return None, None

# Function to iterate over all days and files in a year
def process_year_data(year):
    rain_rate_features = []
    rain_rate_labels = []
    rain_type_features = []
    rain_type_labels = []
    tb1_dir = os.path.join(base_dir, "S1.ABp103-117.GMI.Tc", year)
    tb2_dir = os.path.join(base_dir, "S1.ABp103-117.GMI.TcS2", year)
    vfracConv_dir = os.path.join(base_dir, "S1.ABp083-137.DPRGMI.V06A.9ave.vfracConv", year)
    surfPrecip_dir = os.path.join(base_dir, "S1.ABp083-137.DPRGMI.V06A.9ave.surfPrecipTotRate", year)
    surfTypeIndex_dir = os.path.join(base_dir,"S1.ABp103-117.GMI.surfaceTypeIndex", year)
    year_dir = tb1_dir

    # Iterate over all months in the year directory
    for month_str in sorted(os.listdir(year_dir)):
        # Define directories for each data type for the month
        tb1_month_dir = os.path.join(tb1_dir, month_str)
        tb2_month_dir = os.path.join(tb2_dir, month_str)
        vfracConv_month_dir = os.path.join(vfracConv_dir, month_str)
        surfPrecip_month_dir = os.path.join(surfPrecip_dir, month_str)
        surfTypeIndex_month_dir = os.path.join(surfTypeIndex_dir, month_str)

        # Check if month directories exist
        if not all(os.path.exists(directory) for directory in [tb1_month_dir, tb2_month_dir, vfracConv_month_dir, surfPrecip_month_dir, surfTypeIndex_month_dir]):
            print(f"Not all necessary directories exist for month {month_str}, skipping.")
            continue

        print(f"Processing month: {month_str}")
        
        # Get a list of all days in the tb1_month_dir
        days = [day for day in os.listdir(tb1_month_dir) if os.path.isdir(os.path.join(tb1_month_dir, day))]

        # Process each day
        for day_str in sorted(days):
            day_path = {
                'tb1': os.path.join(tb1_month_dir, day_str),
                'tb2': os.path.join(tb2_month_dir, day_str),
                'vfracConv': os.path.join(vfracConv_month_dir, day_str),
                'surfPrecip': os.path.join(surfPrecip_month_dir, day_str),
                'surfTypeIndex': os.path.join(surfTypeIndex_month_dir, day_str)
            }

            # Check if the day directory exists for all data types
            if not all(os.path.isdir(path) for path in day_path.values()):
                print(f"Directory does not exist for day {day_str} in one or more data types, skipping.")
                continue

            # Assuming all directories exist, list all .npy files for each data type
            files = {datatype: sorted([os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith('.npy')]) 
                     for datatype, dirpath in day_path.items()}

            # Check if there are an equal number of files for each data type
            if not all(len(filelist) == len(files['tb1']) for filelist in files.values()):
                print(f"Unequal number of files across data types for day {day_str}, skipping.")
                continue

            # Process each set of files for the day
            for file_set in zip(files['tb1'], files['tb2'], files['vfracConv'], files['surfPrecip'], files['surfTypeIndex']):
                rain_rate_data, rain_type_data = filter_data(*file_set)
                # Check for consistency in data size before appending
                if rain_rate_data is not None and rain_type_data is not None:
                    if rain_rate_data[0].shape[0] == rain_rate_data[-1].shape[0] and rain_type_data[0].shape[0] == rain_type_data[-1].shape[0]:
                        if rain_rate_data[0].shape[-1] == 104 and rain_type_data[0].shape[-1] == 104:  # Check if the number of features is 104
                            rain_rate_features.append(rain_rate_data[0])
                            rain_rate_labels.append(rain_rate_data[1])
                            rain_type_features.append(rain_type_data[0])
                            rain_type_labels.append(rain_type_data[1])
                        else:
                            print(f"Skipping data with incorrect number of features for day {day_str}")
                    else:
                        print(f"Skipping inconsistent data for day {day_str}")
            print(f"Day {day_str} processing complete.")

    # Stack features and labels for each dataset
    print("Before stacking:")
    num_values_to_print = 100
    print("rain_rate_features:", [x.shape for x in rain_rate_features][:num_values_to_print])
    print("rain_rate_labels:", [x.shape for x in rain_rate_labels][:num_values_to_print])
    print("rain_type_features:", [x.shape for x in rain_type_features][:num_values_to_print])
    print("rain_type_labels:", [x.shape for x in rain_type_labels][:num_values_to_print])
    final_rain_rate_dataset = (np.vstack(rain_rate_features), np.vstack(rain_rate_labels)) if rain_rate_features else (np.array([]), np.array([]))
    final_rain_type_dataset = (np.vstack(rain_type_features), np.vstack(rain_type_labels)) if rain_type_features else (np.array([]), np.array([]))
    print("After stacking:")
    print("final_rain_rate_dataset:", final_rain_rate_dataset[0].shape, final_rain_rate_dataset[1].shape)
    print("final_rain_type_dataset:", final_rain_type_dataset[0].shape, final_rain_type_dataset[1].shape)
    print("All data processed for the year. Compiling final datasets.")
    return final_rain_rate_dataset, final_rain_type_dataset

# Process the data for 2014 and 2015
rain_rate_dataset_2014, rain_type_dataset_2014 = process_year_data("2014")
rain_rate_dataset_2015, rain_type_dataset_2015 = process_year_data("2015")

# Function to concatenate datasets from two years if they are not empty
def concatenate_datasets(dataset_2014, dataset_2015):
    if dataset_2014 and dataset_2015:
        # Concatenating features and labels separately
        features_concatenated = np.vstack((dataset_2014[0], dataset_2015[0]))
        labels_concatenated = np.vstack((dataset_2014[1], dataset_2015[1]))
        return (features_concatenated, labels_concatenated)
    elif dataset_2014:
        return dataset_2014
    elif dataset_2015:
        return dataset_2015
    else:
        print("Both datasets are empty.")
        return (np.array([]), np.array([]))

# Concatenate the datasets for rain rate and rain type separately
final_rain_rate_dataset = concatenate_datasets(rain_rate_dataset_2014, rain_rate_dataset_2015)
final_rain_type_dataset = concatenate_datasets(rain_type_dataset_2014, rain_type_dataset_2015)

# Print the shapes of the final datasets
print("Final Rain Rate Dataset Shapes: Features -", final_rain_rate_dataset[0].shape, ", Labels -", final_rain_rate_dataset[1].shape)
print("Final Rain Type Dataset Shapes: Features -", final_rain_type_dataset[0].shape, ", Labels -", final_rain_type_dataset[1].shape)

# Function to prepare and save datasets
def save_dataset_as_npy(features, labels, save_path):
    # Combine features and labels
    combined_dataset = np.hstack((features, labels))

    # Save the dataset as a .npy file
    np.save(save_path, combined_dataset)
    print(f"Dataset saved at {save_path}.")

# Save the rain rate dataset
save_dataset_as_npy(final_rain_rate_dataset[0], final_rain_rate_dataset[1], '/lustre/home/sasha/GPM/datasets/rain_rate_dataset.npy')

# Save the rain type dataset
save_dataset_as_npy(final_rain_type_dataset[0], final_rain_type_dataset[1], '/lustre/home/sasha/GPM/datasets/rain_type_dataset.npy')

print("Datasets saved as .npy files.")



