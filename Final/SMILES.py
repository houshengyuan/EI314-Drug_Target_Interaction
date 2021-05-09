#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import tensorflow as tf
import keras
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
from collections import defaultdict

def atom_properties(smiles):
    """
    For a SMILES return its all
    :param smiles:
    :return:an encoding vector
    """
    mol = Chem.MolFromSmiles(smiles)
    #calculate amount of atoms(without H)
    natoms = mol.GetNumAtoms()
    #construct adjacency matrix(diagonal set to 1)
    A = GetAdjacencyMatrix(mol)+np.eye(natoms)
    return A


def extract_graph(smiles_list):
    """
    extract the graph structure
    :return:
    """

    pass


class GNN(keras.Model):
   def __init__(self,num_gnn_layers,embedding_size):
     super().__init__()
     self.num_gnn_layers=num_gnn_layers
     self.embedding_size=embedding_size
     self.weight=keras.layers.Dense(units=1, activation='relu',kernel_initializer=tf.truncated_normal_initializer(0.02))
     self.Layers=[self.weight]*self.num_gnn_layers
     self.embedding=keras.layers.Embedding(input_dim=,output_dim=,)


   def gnn(self,x,A):
     for layer in self.Layers:
      hs=keras.activations.relu(layer(x))
      x=x+tf.matmul(A,hs)
     return tf.expand_dims(tf.reduce_mean(x,axis=0),0)


   def call(self, inputs, mask=None):
      atom_properties(inputs)



class GNN1(keras.Model):
   def __init__(self):
     super().__init__()

   def call(self, inputs, mask=None):
      pass


class GNN2(keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs, mask=None):
        pass


class GNN3(keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs, mask=None):
        pass

if __name__=="__main__":
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))