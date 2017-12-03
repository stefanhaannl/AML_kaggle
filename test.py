# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:00:29 2017

@author: Stefan
"""

"""
Main working directory
"""
import preprocessing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DataFile():
    
    def __init__(self,n=0,train=True):
        if train == True:
            self.traindata, self.trainlabels, self.testdata, self.testlabels = self.load_data(n)
            self.inputdimension = list(self.traindata.shape)
            self.outputdimension = list(self.trainlabels.shape)
            self.n_labels = self.outputdimension[1]
        else:
            self.testdata, self.template, self.names = self.load_data(n,train)
        
        
    def load_data(self,n=0,train=True):
        df, template = preprocessing.load_images(n,train)
        df = preprocessing.resize_images(df)
        if train == True:
            df = preprocessing.add_labels(df)
            df = preprocessing.add_label_hotmap(df)
            x_train, x_test, y_train, y_test = preprocessing.split_data(df)
            x_train = np.swapaxes(np.dstack(np.array(x_train)),0,2)
            x_test = np.swapaxes(np.dstack(np.array(x_test)),0,2)
            return x_train, np.array(list(y_train)), x_test, np.array(list(y_test))
        else:
            x_train = df['image_data']
            names = np.array(list(df['image_number']))
            x_train = np.swapaxes(np.dstack(np.array(x_train)),0,2)
            return x_train, template, names
    
    def get_batch(self,batchsize):
        ind = np.random.randint(self.traindata.shape[0],size=batchsize)
        batch_x = self.traindata[ind,:]
        batch_y = self.trainlabels[ind,:]
        return batch_x, batch_y
    
    def get_testbatch(self,batchsize):
        ind = np.random.randint(self.testdata.shape[0],size=batchsize)
        batch_x = self.testdata[ind,:]
        batch_y = self.testlabels[ind,:]
        return batch_x, batch_y