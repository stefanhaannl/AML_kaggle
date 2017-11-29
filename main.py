"""
Main working directory
"""
import preprocessing
import tensorflow as tf
import numpy as np

class AML():
    
    def __init__(self,n=0):
        self.traindata, self.trainlabels = self.load_data(n)
        
    def load_data(self,n=0,train=True):
        df = preprocessing.load_images(n,train)
        df = preprocessing.resize_images(df)
        if train == True:
            df = preprocessing.add_labels(df)
            df = preprocessing.add_label_hotmap(df)
        df = preprocessing.flatten_image_data(df)
        traindata = np.array(df['image_data'].tolist())
        trainlabels = np.array(df['class_hotmap'].tolist())
        return traindata, trainlabels
    
    def get_batch(self,batchsize):
        ind = np.random.randint(self.traindata.shape[0],size=batchsize)
        batch_x = self.traindata[ind,:]
        batch_y = self.trainlabels[ind,:]
        return batch_x, batch_y

        