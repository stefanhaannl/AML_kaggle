"""
Preprocessing file components for loading the information in a workable format.
"""
import numpy as np
import pandas as pd
import os
from PIL import Image
from scipy.misc import imresize

WORKING_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(WORKING_PATH,'data')
TEST_PATH = os.path.join(DATA_PATH,'test')
TRAIN_PATH = os.path.join(DATA_PATH,'train')

def load_images(n=0, train=True):
    """
    Import either test or training images into a pd.dataframe object of np.arrays (images) and label numbers
    """
    im_list = []
    label_list = []

    if train == True:
        path = TRAIN_PATH
    else:
        path = TEST_PATH
    
    for filename in os.listdir(path):
        x = np.array([np.array(Image.open(os.path.join(path,filename)))])
        gray = x[0,:,:] 
        im_list.append(gray)
        id_im = int(filename.split('.')[0])
        label_list.append(id_im)
        n-=1
        if n == 0:
            break

    data_dict = {"image_data":im_list,"image_number":label_list}
    df = pd.DataFrame(data_dict)
    return df

def normalize_images(images):
    """
    Squeezes the image data into the average size of the image data-set. All images will be transformed into the same width and height
    """
    sizes = pd.DataFrame()
    sizes['WH'] = images['image_data'].apply(lambda x: np.shape(x))
    sizes['W'] = sizes['WH'].apply(lambda x:x[0])
    sizes['H'] = sizes['WH'].apply(lambda x:x[1])
    means = sizes[['W','H']].mean()
    new_sizes = (int(round(means['W'])),int(round(means['H'])))
    images['image_data'] = images['image_data'].apply(lambda x: imresize(x,new_sizes))
    return images

def add_labels(images):
    """
    Add lables to Training images
    Transform pd.dataframe object of np.arrays (images) into pd.series of images and text labels
    """
    labels = pd.read_csv(os.path.join(DATA_PATH,'train_onelabel.csv'))
    labels.columns = ['image_number','class_id']
    labels['image_number'] = labels['image_number'].apply(lambda x:int(x.split('.')[0]))
    required_labels = labels[labels.image_number.isin(list(images['image_number']))]
    return pd.merge(images,required_labels,on='image_number')
    
