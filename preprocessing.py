"""
Preprocessing file for loading the information in a workable format.
"""
import numpy as np
import pandas as pd
import os
from PIL import Image

WORKING_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(WORKING_PATH,'data')

def load_images(n=0, train=True):
    """
    Import either test or training images into a pd.dataframe object of np.arrays (images) and label numbers
    """
    im_list = []
    label_list = []

    path = ""
    if train == True:
        path = "train"
    else:
        path = "test"
    path = os.path.join(DATA_PATH,path)
    for filename in os.listdir(path):
        x = np.array([np.array(Image.open(os.path.join(path,filename)))])
        im_list.append(x)
        id_im = int(filename.split('.')[0])
        label_list.append(id_im)
        n-=1
        if n == 0:
            break

    data_dict = {"image_data":im_list,"image_number":label_list}
    df = pd.DataFrame(data_dict)
    return df

def add_labels(images):
    """
    Add lables to Training images
    Transform pd.dataframe object of np.arrays (images) into pd.series of images and text labels
    """
    labels = pd.read_csv(os.path.join(DATA_PATH,'train_onelabel.csv'))
    labels.columns = ['image_number','class']
    labels['image_number'] = labels['image_number'].apply(lambda x:int(x.split('.')[0]))
    required_labels = labels[labels.image_number.isin(list(images['image_number']))]
    return pd.merge(images,required_labels,on='image_number')
    
