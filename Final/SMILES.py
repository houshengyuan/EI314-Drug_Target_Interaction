#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import tensorflow as tf
import keras
from rdkit import Chem
from rdkit.Chem import AllChem

def atom_properties(smiles):
    """
    For a SMILES return its all
    :param smiles:
    :return:an encoding vector
    """
    mol = Chem.MolFromSmiles(smiles)

    mol.get


def extract_graph(smiles_list):
    """
    extract the graph structure
    :return:
    """

    pass


class GNN(keras.Model):
   def __init__(self):
     super().__init__()

   def call(self, inputs, mask=None):
      pass


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

