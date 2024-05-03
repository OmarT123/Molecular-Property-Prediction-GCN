from rdkit import Chem
import numpy as np
import pandas as pd
import sys
import os
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

MAX_NUM_ATOMS = 9

PAULING_ELECTRONEGATIVITY = {
    6: 2.55,   # Carbon
    7: 3.04,   # Nitrogen
    8: 3.44,   # Oxygen
}

def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    feature_matrix = []

    for atom in mol.GetAtoms():
        feature_matrix.append(atom_feature(atom))

    while (len(feature_matrix) < MAX_NUM_ATOMS):
        feature_matrix.append(np.zeros(21, dtype=int))
    
    return feature_matrix

def calculate_molecule_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    feature_matrix = np.zeros((MAX_NUM_ATOMS, 8), dtype=int)  # Modified to include electronegativity

    for i, atom in enumerate(mol.GetAtoms()):
        # Atomic number
        atomic_number = atom.GetAtomicNum()
        feature_matrix[i, 0] = atomic_number
        # Atomic mass
        feature_matrix[i, 1] = atom.GetMass()

        # Atomic charge
        # feature_matrix[i, 2] = atom.GetFormalCharge()

        # Atomic hybridization
        hybridization = str(atom.GetHybridization()).lower()
        if hybridization == 's':
            feature_matrix[i, 2] = 0
        elif hybridization == 'sp':
            feature_matrix[i, 2] = 1
        elif hybridization == 'sp2':
            feature_matrix[i, 2] = 2
        elif hybridization == 'sp3':
            feature_matrix[i, 2] = 3
        else:
            feature_matrix[i, 2] = -1  # Unknown hybridization

        # Bond types
        bond_types = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
        feature_matrix[i, 3] = sum(bond_types)

        # Molecular connectivity
        feature_matrix[i, 4] = len(atom.GetNeighbors())

        # Aromaticity
        # feature_matrix[i, 6] = int(atom.GetIsAromatic())
        
        # Electronegativity (Pauling scale)
        feature_matrix[i, 5] = PAULING_ELECTRONEGATIVITY.get(atomic_number, 0) 
    i+=1
    # Fill remaining rows with zeros if less than MAX_NUM_ATOMS
    while i < MAX_NUM_ATOMS:
        feature_matrix[i, :] = 0.0
        i += 1
    
    return feature_matrix


def adjacency_matrix_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    adj_matrix = np.zeros((MAX_NUM_ATOMS, MAX_NUM_ATOMS), dtype=int)

    # Loop over bonds and update adjacency matrix
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()  # Get bond type (1.0 for single bond, 2.0 for double bond, etc.)
        adj_matrix[begin_idx, end_idx] = bond_type
        adj_matrix[end_idx, begin_idx] = bond_type

    return adj_matrix

def atom_feature(atom): # Computes features of each atom 
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set): # One hot encoding function
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set): # One hot encoding function
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def convert_dataset(dataset_path, stage):
    dataset = pd.read_csv(dataset_path)
    molecules = dataset['smiles']

    features = []
    adjs = []
    total_errors = 0

    for mol in molecules:
        try:
            adj = adjacency_matrix_from_smiles(mol)
            feature = mol_features(mol)
            adjs.append(adj)
            features.append(feature)
        except Exception as e:
            total_errors += 1
            print(mol)
            print(e)


    print(f"Successfully converted {len(molecules) - total_errors} molecules out of {len(molecules)} to graphs for {stage} stage")

    adjs = np.array(adjs)
    features = np.array(features)

    dir_path = os.path.dirname(dataset_path)

    np.save(os.path.join(dir_path, f'{stage}_adj.npy'), adjs)
    np.save(os.path.join(dir_path, f'{stage}_features.npy'), features)

convert_dataset('./database/QM9_deepchem/train_data.csv', 'train')
convert_dataset('./database/QM9_deepchem/val_data.csv', 'valid')
convert_dataset('./database/QM9_deepchem/test_data.csv', 'test')
