#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import tensorflow as tf
import keras
#自己定义，以下是一个模板，不一定非要用CNN

class Protein_CNN(keras.Model):
   def __init__(self):
      super(Protein_CNN).__init__(self)

   def call(self,inputs,training=None,mask=None):
       pass


if __name__ == "main":
    pass
