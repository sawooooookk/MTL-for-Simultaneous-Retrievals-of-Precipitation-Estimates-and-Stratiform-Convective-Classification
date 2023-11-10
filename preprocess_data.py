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
    # Ensure that tb_diffs is a 1D array
    tb_diffs = np.array([tb_array[center_pixel_index, i] - tb_array[center_pixel_index, j] 
                         for i in range(12) for j in range(i+1, 13)]).flatten()

    # Last 13 features: Standard deviation of TB for the 2 surrounding pixels
    surrounding_indices = [(center_pixel_index-1), (center_pixel_index+1)]
    tb_stds = np.array([np.std(tb_array[surrounding_indices, i]) for i in range(13)]).flatten()

    # Combine all features into a single array
    features = np.concatenate((tb_center, tb_diffs, tb_stds))
    # print(f"Features shape: {features.shape}")
    return features



# Function to load and filter data for one .npy file
def filter_data(tb1_file, tb2_file, vfracConv_file, surfPrecip_file, surfTypeIndex_file):
    # print(f"Processing files: {tb1_file}, {tb2_file}, {vfracConv_file}, {surfPrecip_file}, {surfTypeIndex_file}")
    # Load the data
    tb1_data = np.load(tb1_file)
    tb2_data = np.load(tb2_file)
    convective_fraction_data = np.load(vfracConv_file)
    rain_rate_data = np.load(surfPrecip_file)
    surface_type_data = np.load(surfTypeIndex_file)
    
    # Assume center pixel index is 27 for the cropped 083-137 arrays
    center_pixel_index = 29

    # Assume center pixel index is 27 for the cropped 103-117 arrays
    center_pixel_index2 = 9
    
    # Find indices where the center pixel rain rate is not 0.0
    valid_precip_indices = rain_rate_data[:, center_pixel_index] > 0.1
    valid_surface_type_indices = surface_type_data[:, center_pixel_index2] == 1

    # Find indices where both the center pixel rain rate is not 0.0 and the center pixel is over ocean
    valid_indices = np.logical_and(valid_precip_indices, valid_surface_type_indices)
    
    if convective_fraction_data.size == 0:
        print("Warning: Convective fraction data array is empty.")
        return None

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
    
    # Stack the labels side by side with one-hot encoding for convective fraction
    convective_fraction_labels = convective_fraction_data[:, center_pixel_index].reshape(-1, 1)
    convective_fraction_labels = convective_fraction_labels.squeeze()

    if convective_fraction_labels.size > 0:
        convective_fraction_one_hot = np.zeros((convective_fraction_labels.shape[0], 2))
        # Stratiform in the first column if convective fraction < 0.5
        convective_fraction_one_hot[convective_fraction_labels < 0.5, 0] = 1
        # Convective in the second column if convective fraction >= 0.5
        convective_fraction_one_hot[convective_fraction_labels >= 0.5, 1] = 1
        # Combine one-hot encoded classification labels with rain rate labels
        rain_rate_labels = rain_rate_data[:, center_pixel_index].reshape(-1, 1)    
    else:
        print("Warning: No valid convective fraction labels found.")
        return None
    
    # After creating one-hot encoded labels
    stratiform_indices = np.where(convective_fraction_labels < 0.5)[0]
    convective_indices = np.where(convective_fraction_labels >= 0.5)[0]

    # Calculate the number of stratiform samples to keep (55% of the original)
    num_stratiform_to_keep = int(len(stratiform_indices) * 0.55)

    # Randomly select stratiform indices to keep
    np.random.seed(42)  # For reproducibility
    stratiform_indices_to_keep = np.random.choice(stratiform_indices, num_stratiform_to_keep, replace=False)

    # Combine the selected stratiform indices with all convective indices
    selected_indices = np.concatenate((stratiform_indices_to_keep, convective_indices))

    # Filter all data based on the selected indices
    tb_features = tb_features[selected_indices]
    convective_fraction_one_hot = convective_fraction_one_hot[selected_indices]
    rain_rate_labels = rain_rate_labels[selected_indices]

    # Stack the labels side by side
    labels = np.hstack((convective_fraction_one_hot, rain_rate_labels))
    
    if tb_features.size == 0 or labels.size == 0 or tb_features.shape[0] != labels.shape[0]:
        print("Warning: Data arrays are empty or do not match in size.")
        print(f"TB features shape: {tb_features.shape}")
        print(f"Labels shape: {labels.shape}")
        # Handle this situation appropriately, such as by skipping this set of files
        return None

    # Combine the TB features with the labels
    combined_data = np.hstack((tb_features, labels))
    
    # print("Filtering complete.")
    return combined_data

# Function to iterate over all days and files in a year
def process_year_data(year):
    complete_dataset = []
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
                daily_data = filter_data(*file_set)
                if daily_data is not None:
                    complete_dataset.append(daily_data)

            print(f"Day {day_str} processing complete.")

    if not complete_dataset:  # If the list is empty, no data was processed
        print("No data processed for the year.")
        return np.array([])  # Return an empty array if no data was processed

    # Concatenate all daily data into a single dataset
    final_dataset = np.vstack(complete_dataset)
    print("All data processed for the year. Compiling final dataset.")
    return final_dataset

# Process the data for 2014 and 2015
dataset_2014 = process_year_data("2014")
dataset_2015 = process_year_data("2015")

# Concatenate the data from both years if they are not empty
if dataset_2014.size > 0 and dataset_2015.size > 0:
    dataset = np.vstack((dataset_2014, dataset_2015))
    print(f"Combined dataset shape: {dataset.shape}")
else:
    print("One of the datasets is empty, cannot concatenate.")
    dataset = dataset_2014 if dataset_2014.size > 0 else dataset_2015

# Convert to a PyTorch tensor and save as a .pt file, if the combined dataset is not empty
if dataset.size > 0:
    dataset = torch.tensor(dataset, dtype=torch.float32)
    print(f"Combined dataset for 2014 and 2015 created. Shape: {dataset.shape}")
else:
    print("No data available to create the combined dataset.")

# Assuming 'dataset' is a tensor where the last 3 columns are the labels
features = dataset[:, :104]
labels = dataset[:, 104:]

# Split the dataset into training and testing sets
train_size = int(0.8 * len(features))
test_size = len(features) - train_size

# Randomly split the dataset into training and testing
train_features, test_features = random_split(features, [train_size, test_size])
train_labels, test_labels = random_split(labels, [train_size, test_size])

# Save the datasets as tuples
torch.save((train_features, train_labels), '/lustre/home/sasha/GPM/train1.pt')
torch.save((test_features, test_labels), '/lustre/home/sasha/GPM/test1.pt')


print("Train and test datasets saved.")
# You can now use 'train_dataset' and 'test_dataset' for your model training and evaluation


