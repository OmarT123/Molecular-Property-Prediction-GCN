import numpy as np
import sys
from rdkit import Chem
import pandas as pd


def adj_k(adj, k): # Compute k-th power of adj matrix

    ret = adj
    for i in range(0, k-1):
        ret = np.dot(ret, adj)  

    return convertAdj(ret)

def convertAdj(adj): # Convert adj matrix into a binary matrix

    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim*dim)
    c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
    d = c.reshape((dim, dim))

    return d

def convertToGraph(smiles_list, k): # Convert a list of smiles text to graph representation
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 50
    for i in smiles_list:
        # Mol
        iMol = Chem.MolFromSmiles(i.strip())
        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if( iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            iFeature = np.zeros((maxNumAtoms, 58))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only
            iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(adj_k(np.asarray(iAdj), k))
    features = np.asarray(features)

    return adj, features
    
def atom_feature(atom): # Computes features of each atom
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)

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

smiles = "C(=O)NC=O"
mol = Chem.MolFromSmiles(smiles)
for atom in mol.GetAtoms():
    print( atom_feature(atom).shape)
    break

# execution : 
# Re-use several scripts in https://github.com/HIPS/neural-fingerprint
# python smilesToGraph.py CEP 1000 1    --- to generate graph inputs for CEP dataset 
# python smilesToGraph.py ZINC 10000 1  --- to generate graph inputs for ZINC dataset

# dbName = sys.argv[1]        # CEP, ZINC, QM9
# length = int(sys.argv[2])   # num molecules in graph input files
# k = int(sys.argv[3])        # neighbor distance


# dbName = "QM9_deepchem"
# dbPath = "./QM9_deepchem/qm9.csv"
# length = 500
# k = 1
# df = pd.read_csv(dbPath)
# smiles_list = df['smiles'].tolist()
# num_smiles = len(smiles_list)

# # smiles_f = open('./'+dbName+'/smiles.txt')
# # smiles_list = smiles_f.readlines()
# # print (len(smiles_list))
# maxNum = int(num_smiles/length)

# for i in range(maxNum+1):
#     lb = i*length
#     ub = (i+1)*length
#     adj, features = convertToGraph(smiles_list[lb:ub], k)
#     print (np.asarray(features).shape)
#     np.save('./'+dbName+'/adj/'+str(i)+'.npy', adj)
#     np.save('./'+dbName+'/features/'+str(i)+'.npy', features)