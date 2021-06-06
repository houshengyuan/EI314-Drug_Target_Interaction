import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
import torch
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
import os
import pickle

MAX_ATOM = 800
MAX_BOND = MAX_ATOM * 2
RATIO = int(100000 / 15000)
ELEM_LIST = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'B', 'H', 'Ca', 'Zr', 'Si', 'Dy', 'Pb', 'V', 'Ti', 'Sr',
             'Bi', 'Pd', 'Al', 'Yb', 'Mn', 'Tl', 'As', 'Mo', 'Fe',
             'Sn', 'Ru', 'K', 'Pt', 'Li', 'Ag', 'Au', 'Sb', 'Cd', 'Mg',
             'Cu', 'Cr', 'Be', 'Nd', 'Co', 'I', 'Ba', 'Ge', 'Eu',
             'Zn', 'In', 'Hg', 'Na', 'Se', 'Sc', 'Ni']
ATOM_FDIM = len(ELEM_LIST) + 8 + 5 + 4 + 5 + 4 + 1 + 1 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
              'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
               '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
               'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
               'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
               'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100


def read_file_training_dataset_drug_target_pairs(path):
    file = open(path, "r")
    X_drug = []
    X_target = []
    y = []
    for i, aline in enumerate(file):
        if i != 0:
            values = aline.strip('\n').split(',')
            X_drug.append(values[0])
            X_target.append(values[1])
            y.append(float(values[2]))
    return np.array(X_drug), np.array(X_target), np.array(y)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not (mol is None):
        Chem.Kekulize(mol)
    return mol


def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return torch.Tensor(onehot_encoding(atom.GetSymbol(), ELEM_LIST)
                        + onehot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7])
                        + onehot_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                        + onehot_encoding(int(atom.GetChiralTag()), [0, 1, 2, 3])
                        + onehot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
                        + onehot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3])
                        + [atom.GetIsAromatic()] + [atom.GetHybridization()] + [atom.IsInRing()])


def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt ==
             Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onehot_encoding(stereo, [0, 1, 2, 3, 4, 5])
    return torch.Tensor(fbond + fstereo)


def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)


def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


def smiles2mpnnfeature(smiles):
    try:
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
        fatoms, fbonds = [], [padding]
        in_bonds, all_bonds = [], [(-1, -1)]
        mol = get_mol(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom))
            in_bonds.append([])
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx()
            y = a2.GetIdx()
            b = len(all_bonds)
            all_bonds.append((x, y))
            fbonds.append(torch.cat([fatoms[x], bond_features(bond)], 0))
            in_bonds[y].append(b)
            b = len(all_bonds)
            all_bonds.append((y, x))
            fbonds.append(torch.cat([fatoms[y], bond_features(bond)], 0))
            in_bonds[x].append(b)
        total_bonds = len(all_bonds)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(n_atoms, MAX_NB).float()
        bgraph = torch.zeros(total_bonds, MAX_NB).float()
        for a in range(n_atoms):
            for i, b in enumerate(in_bonds[a]):
                agraph[a, i] = b
        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):
                if all_bonds[b2][0] != y:
                    bgraph[b1, i] = b2
    except:
        fatoms = torch.zeros(0, 82)
        fbonds = torch.zeros(0, 93)
        agraph = torch.zeros(0, 6)
        bgraph = torch.zeros(0, 6)
    Natom, Nbond = fatoms.shape[0], fbonds.shape[0]
    atoms_completion_num = MAX_ATOM - fatoms.shape[0]
    bonds_completion_num = MAX_BOND - fbonds.shape[0]
    fatoms_dim = fatoms.shape[1]
    fbonds_dim = fbonds.shape[1]
    fatoms = torch.cat([fatoms, torch.zeros(atoms_completion_num, fatoms_dim)], 0)
    fbonds = torch.cat([fbonds, torch.zeros(bonds_completion_num, fbonds_dim)], 0)
    agraph = torch.cat([agraph.float(), torch.zeros(atoms_completion_num, MAX_NB)], 0)
    bgraph = torch.cat([bgraph.float(), torch.zeros(bonds_completion_num, MAX_NB)], 0)
    shape_tensor = torch.Tensor([Natom, Nbond]).view(1, -1)
    return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor.float()]


def create_fold(df, fold_seed, frac,aug):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac=(0 if val_frac==0 else val_frac / (1 - test_frac)), replace=False, random_state=1)
    train = train_val[~train_val.index.isin(val.index)]
    train.reset_index(drop=True, inplace=True)
    if aug:
        x = len(train)
        for i in range(x):
            if train.loc[i]['Label'] == 1.0:
                a = train.loc[i]
                d = pd.DataFrame(a).T
                train = train.append([d] * RATIO)
        train = train.sample(frac=1, replace=True, random_state=1)
        train.reset_index(drop=True,inplace=True)
    return train, val, test


def encode_drug(df_data, column_name='SMILES', save_column_name='SMILES'):
    print('Encoding Drug By MPNN ing...')
    unique = pd.Series(df_data[column_name].unique()).apply(smiles2mpnnfeature)
    unique_dict = dict(zip(df_data[column_name].unique(), unique))
    df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
    return df_data


def encode_protein(df_data, column_name='FASTA', save_column_name='FASTA'):
    print('Encoding Protein By CNN ing...')
    cnn = pd.Series(df_data[column_name].unique()).apply(trans_protein)
    cnn_dict = dict(zip(df_data[column_name].unique(), cnn))
    df_data[save_column_name] = [cnn_dict[i] for i in df_data[column_name]]
    return df_data


def data_process(X_drug=None, X_target=None, y=None, frac=[0.8, 0.1, 0.1],
                 random_seed=1, aug=True):
    if isinstance(X_target, str):
        X_target = [X_target]
    df_data = pd.DataFrame(zip(X_drug, X_target, y))
    df_data.rename(columns={0: 'SMILES', 1: 'FASTA', 2: 'Label'}, inplace=True)
    df_data = encode_drug(df_data)
    df_data = encode_protein(df_data)
    train, val, test = create_fold(df_data, random_seed, frac,aug=aug)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class data_loader(data.Dataset):
    def __init__(self, list_IDs, labels, df, **config):
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df
        self.config = config

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        v_p = protein_2_embed(self.df.iloc[index]['FASTA'])
        y = self.labels[index]
        return v_d, v_p, y



def trans_protein(x):
    temp = list(x.upper())
    temp = [i if i in amino_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_PROTEIN:
        temp = temp + ['?'] * (MAX_SEQ_PROTEIN - len(temp))
    else:
        temp = temp[:MAX_SEQ_PROTEIN]
    return temp


def protein_2_embed(x):
    return enc_protein.transform(np.array(x).reshape(-1, 1)).toarray().T


def save_dict(path, obj):
    with open(os.path.join(path, 'config.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(path):
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        obj = pickle.load(f)
        return obj


def mpnn_feature_collate_func(x):
    N_atoms_scope = torch.cat([i[4] for i in x], 0)
    f_a = torch.cat([x[j][0].unsqueeze(0) for j in range(len(x))], 0)
    f_b = torch.cat([x[j][1].unsqueeze(0) for j in range(len(x))], 0)
    agraph_lst, bgraph_lst = [], []
    for j in range(len(x)):
        agraph_lst.append(x[j][2].unsqueeze(0))
        bgraph_lst.append(x[j][3].unsqueeze(0))
    agraph = torch.cat(agraph_lst, 0)
    bgraph = torch.cat(bgraph_lst, 0)
    return [f_a, f_b, agraph, bgraph, N_atoms_scope]


# utils.smiles2mpnnfeature -> utils.mpnn_collate_func -> utils.mpnn_feature_collate_func -> encoders.MPNN.forward
def mpnn_collate_func(x):
    mpnn_feature = [i[0] for i in x]
    mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
    x_remain = [list(i[1:]) for i in x]
    x_remain_collated = default_collate(x_remain)
    return [mpnn_feature] + x_remain_collated
