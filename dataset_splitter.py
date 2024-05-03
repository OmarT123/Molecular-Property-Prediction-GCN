import os
import sys
import pandas as pd
import numpy as np


def shuffle_and_split_dataset(dataset_path):
    dataset = pd.read_csv(os.path.join(dataset_path, 'qm9.csv'))

    shuffled_data = dataset.sample(frac=1, random_state=42)
    
    train_size = 107_100 #int((len(dataset)*0.8)/10)*10
    val_size = 13_385 #int((0.1 * len(dataset))/10)*10

    training_dataset = shuffled_data[0: train_size]
    validation_dataset = shuffled_data[train_size: train_size + val_size]
    test_dataset = shuffled_data[train_size + val_size : ]

    # Adds 15 random rows again at the end of the validation dataset to get a size divisible by 100
    additional_validation_rows = validation_dataset.sample(n=15, random_state=42)
    validation_dataset = pd.concat([validation_dataset, additional_validation_rows])

    training_dataset.to_csv(os.path.join(dataset_path, 'train_data.csv'), index=False)
    validation_dataset.to_csv(os.path.join(dataset_path, 'val_data.csv'), index=False)
    test_dataset.to_csv(os.path.join(dataset_path, 'test_data.csv'), index=False)

shuffle_and_split_dataset('./database/QM9_deepchem')
