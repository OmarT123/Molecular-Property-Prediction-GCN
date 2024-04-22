import os
import shutil
import random

def split_dataset(dataset_path, training_idx, validation_idx):
    # Create directories for training, validation, and testing sets
    training_path = os.path.join(dataset_path, 'training_dataset')
    validation_path = os.path.join(dataset_path, 'validation_dataset')
    testing_path = os.path.join(dataset_path, 'testing_dataset')

    adj_path = os.path.join(dataset_path, 'adj')
    features_path = os.path.join(dataset_path, 'features')

    os.makedirs(os.path.join(training_path, 'adj'), exist_ok=True)
    os.makedirs(os.path.join(training_path, 'features'), exist_ok=True)

    os.makedirs(os.path.join(validation_path,'adj'), exist_ok=True)
    os.makedirs(os.path.join(validation_path, 'features'), exist_ok=True)

    os.makedirs(os.path.join(testing_path, 'adj'), exist_ok=True)
    os.makedirs(os.path.join(testing_path, 'features'), exist_ok=True)

    # Get list of adjacency lists files and randomize them
    adj_files = os.listdir(adj_path)
    random.shuffle(adj_files)  

    # Divide adj files into training, validation, and testing datasets
    training_files = adj_files[:training_idx]
    validation_files = adj_files[training_idx: validation_idx]
    testing_files = adj_files[validation_idx:]
    
    # Move files to new folders
    for file_name in training_files:
        shutil.copy(os.path.join(adj_path, file_name), os.path.join(training_path, 'adj', file_name))
        shutil.copy(os.path.join(features_path, file_name), os.path.join(training_path, 'features', file_name))
    for file_name in validation_files:
        shutil.copy(os.path.join(adj_path, file_name), os.path.join(validation_path, 'adj', file_name))
        shutil.copy(os.path.join(features_path, file_name), os.path.join(validation_path, 'features', file_name))
    for file_name in testing_files:
        shutil.copy(os.path.join(adj_path, file_name), os.path.join(testing_path, 'adj', file_name))
        shutil.copy(os.path.join(features_path, file_name), os.path.join(testing_path, 'features', file_name))

    print("Files divided into training, validation, and testing sets successfully!")

split_dataset('./database/QM9_deepchem', 214, 241)