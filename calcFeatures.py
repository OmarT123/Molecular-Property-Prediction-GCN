# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import Descriptors


# def get_atom_coordinates(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None

#     # mol = Chem.AddHs(mol)  # Add hydrogens for better representation of 3D structure
#     AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
#     AllChem.MMFFOptimizeMolecule(mol)  # Optimize geometry using MMFF force field

#     conformer = mol.GetConformer()
#     num_atoms = mol.GetNumAtoms()
#     coordinates = np.zeros((num_atoms, 3))

#     for atom in mol.GetAtoms():
#         atom_idx = atom.GetIdx()
#         pos = conformer.GetAtomPosition(atom_idx)
#         coordinates[atom_idx] = [pos.x, pos.y, pos.z]
#     return coordinates


# def get_atomic_numbers(smiles):
#     """
#     Calculate atomic numbers from a molecule in SMILES format.
    
#     Args:
#         smiles (str): SMILES representation of the molecule.
    
#     Returns:
#         list: List of atomic numbers.
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
#     return np.array(atomic_numbers, dtype=float)

# def get_atomic_charges(smiles):
#     """
#     Calculate atomic charges from a molecule in SMILES format.
    
#     Args:
#         smiles (str): SMILES representation of the molecule.
    
#     Returns:
#         list: List of atomic charges.
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
#     return np.array(charges, dtype=float)

# def get_atomic_masses(smiles):
#     """
#     Calculate atomic masses from a molecule in SMILES format.
    
#     Args:
#         smiles (str): SMILES representation of the molecule.
    
#     Returns:
#         list: List of atomic masses.
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     masses = [atom.GetMass() for atom in mol.GetAtoms()]
#     return masses

# def get_atomic_hybridizations(smiles):
#     """
#     Calculate atomic hybridizations from a molecule in SMILES format.

#     Args:
#         smiles (str): SMILES representation of the molecule.

#     Returns:
#         list: List of atomic hybridizations (numerical representation).
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     hybridizations = [atom.GetHybridization().real for atom in mol.GetAtoms()]  # Get numerical representation
#     return np.array(hybridizations, dtype=float)

# def get_bond_types(smiles):
#     """
#     Calculate bond types from a molecule in SMILES format.

#     Args:
#         smiles (str): SMILES representation of the molecule.

#     Returns:
#         list: List of bond types (numerical representation).
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     bond_types = [bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]  # Get numerical representation
#     return bond_types

# def get_molecular_connectivity(smiles):
#     """
#     Calculate molecular connectivity from a molecule in SMILES format.

#     Args:
#         smiles (str): SMILES representation of the molecule.

#     Returns:
#         float: Molecular connectivity value.
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     mc = Descriptors.MolWt(mol)  # Use molecular weight as a measure of molecular connectivity
#     return np.array([mc])



# def combine_features(smiles):
#     """
#     Combine all features into a single feature matrix.

#     Args:
#         smiles (str): SMILES representation of the molecule.

#     Returns:
#         np.array: Feature matrix containing all features (shape: (9, 10)).
#     """
#     # Initialize an empty feature matrix
#     feature_matrix = np.zeros((9, 10))

#     # Get features
#     atomic_coords = get_atom_coordinates(smiles)
#     atomic_nums = get_atomic_numbers(smiles)
#     atomic_charges = get_atomic_charges(smiles)
#     atomic_masses = get_atomic_masses(smiles)
#     atomic_hybridizations = get_atomic_hybridizations(smiles)
#     bond_types = get_bond_types(smiles)
#     molecular_connectivity = get_molecular_connectivity(smiles)

#     # Determine the number of atoms in the molecule
#     num_atoms = len(atomic_coords)
    
#     # Update feature matrix with features for each atom
#     for i in range(num_atoms):
#         feature_matrix[i, 0:3] = atomic_coords[i] if i < len(atomic_coords) else 0 #matrix
#         feature_matrix[i, 3] = atomic_nums[i] if i < len(atomic_nums) else 0 #list
#         feature_matrix[i, 4] = atomic_charges[i] if i < len(atomic_charges) else 0 #list
#         feature_matrix[i, 5] = atomic_masses[i] if i < len(atomic_masses) else 0 #list
#         feature_matrix[i, 6] = atomic_hybridizations[i] if i < len(atomic_hybridizations) else 0 #list
#         feature_matrix[i, 7] = bond_types[i] if i < len(bond_types) else 0 #list
#         feature_matrix[i, 8] = molecular_connectivity[i] if i < len(molecular_connectivity) else 0 #float

#     return feature_matrix



from rdkit import Chem
import numpy as np
import pandas as pd
import sys

def calculate_molecule_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    feature_matrix = np.zeros((9, 7), dtype=float)

    for i, atom in enumerate(mol.GetAtoms()):
        # Atomic number
        feature_matrix[i, 0] = atom.GetAtomicNum()

        # Atomic mass
        feature_matrix[i, 1] = atom.GetMass()

        # Atomic charge
        feature_matrix[i, 2] = atom.GetFormalCharge()

        # Atomic hybridization
        hybridization = str(atom.GetHybridization()).lower()
        if hybridization == 's':
            feature_matrix[i, 3] = 0
        elif hybridization == 'sp':
            feature_matrix[i, 3] = 1
        elif hybridization == 'sp2':
            feature_matrix[i, 3] = 2
        elif hybridization == 'sp3':
            feature_matrix[i, 3] = 3
        else:
            feature_matrix[i, 3] = -1  # Unknown hybridization

        # Bond types
        bond_types = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
        feature_matrix[i, 4] = sum(bond_types)

        # Molecular connectivity
        feature_matrix[i, 5] = len(atom.GetNeighbors())

        # Aromaticity
        feature_matrix[i, 6] = int(atom.GetIsAromatic())
    while i < 9:
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

MAX_NUM_ATOMS = 9
molecules = pd.read_csv('./database/QM9_deepchem/qm9.csv')['smiles']

features = []
tmp_features = []
adjs = []
tmp_adj = []
i = 1
total_errors = 0

# for mol in molecules:
#     try:
#         adj = adjacency_matrix_from_smiles(mol)
#         feature = calculate_molecule_features(mol)
#         tmp_adj.append(adj)
#         tmp_features.append(feature)
#         if i % 500 == 0:
#             adjs.append(adj)
#             features.append(tmp_features)
#             tmp_adj = []
#             tmp_features = []
#             print(adjs.shape)
#         i += 1
#     except:
#         total_errors += 1

for mol in molecules:
    try:
        adj = adjacency_matrix_from_smiles(mol)
        feature = calculate_molecule_features(mol)
        adjs.append(adj)
        features.append(feature)
    except:
        total_errors += 1

print(f"Successfully converted {len(molecules) - total_errors} molecules out of {len(molecules)}")

# if len(tmp_features) > 0:
#     while len(tmp_features) != 500:
#         tmp_adj.append(np.zeros(9, 7), dtype=float)
#         tmp_features.append(np.zeros((9, 7), dtype=float))
#     adjs.append(tmp_adj)
#     features.append(tmp_features)

# if len(tmp_adj) > 0:
#     adjs.append(tmp_adj)
#     features.append(tmp_features)

adjs = np.array(adjs)
features = np.array(features)

np.save('./database/QM9_deepchem/new_adj.npy', adjs)
np.save('./database/QM9_deepchem/new_features.npy', features)
# print(features)
# print(features.shape)
