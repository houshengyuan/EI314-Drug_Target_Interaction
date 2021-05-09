#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import tensorflow as tf
import keras
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
from collections import defaultdict


atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
word_dict = defaultdict(lambda: len(word_dict))


def atom_A(smiles):
    """
    For a SMILES sequence return its adjacent matrix ( with diagonal 1)
    """
    mol = Chem.MolFromSmiles(smiles)
    #calculate amount of atoms(without H)
    natoms = mol.GetNumAtoms()
    #construct adjacency matrix(diagonal set to 1)
    A = GetAdjacencyMatrix(mol)+np.eye(natoms)
    return A


def atom_property(smiles):
    """
    return the property of the input atom,including
    atom symbol id, degree, totalnumH, ImplicitValence,
    IsAromatic, Hybridization, IsinRing (and so on......)
    """
    mol = Chem.MolFromSmiles(smiles)
    atom_set = mol.GetAtoms()
    atoms = [x.GetSymbol() for x in atom_set]
    atom_symbol=[atom_dict[x] for x in atoms]
    atom_numH= [x.GetTotalNumHs() for x in atom_set]
    atom_degree = [x.GetDegree() for x in atom_set]
    atom_valence = [x.GetImplicitValence() for x in atom_set]
    atom_aromatic = [x.GetIsAromatic() for x in atom_set]
    atom_hybridization = [x.GetHybridization() for x in atom_set]
    atom_IsInRing = [x.IsInRing() for x in atom_set]
    return np.array(atom_symbol),np.array(atom_numH),np.array(atom_degree),np.array(atom_valence),\
           np.array(atom_aromatic),np.array(atom_hybridization),np.array(atom_IsInRing)


def bond_properties(smiles):
    pass


def extract_fingerprints():
    pass


def extract_graph(smiles_list):
    """
    extract the graph structure
    :return:
    """

    pass


class GNN(keras.Model):
   def __init__(self,num_gnn_layers=10,embedding_size=53):
     super(GNN,self).__init__()
     self.num_gnn_layers=num_gnn_layers
     self.embedding_size=embedding_size
     self.weight=keras.layers.Dense(units=self.embedding_size, activation='relu',kernel_initializer=tf.truncated_normal_initializer(0.02))
     self.Layers=[self.weight]*self.num_gnn_layers
     #self.embedding=keras.layers.Embedding(input_dim=,output_dim=self.embedding_size)


   def gnn(self,x,A):
     for layer in self.Layers:
      hs=keras.activations.relu(layer(x))
      x=x+tf.matmul(A,hs)
     return tf.expand_dims(tf.reduce_mean(x,axis=0),0)


   def call(self, inputs, mask=None):
      atom_A(inputs)


if __name__=="__main__":
    smiles="COC1=C(OC)C=C2C(N)=NC(=NC2=C1)N1CCN(CC1)C(=O)C1CCCO1"
    atom_property(smiles)