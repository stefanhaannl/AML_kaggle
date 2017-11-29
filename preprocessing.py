"""
Preprocessing file components for loading the information in a workable format.
"""
import numpy as np
import pandas as pd
import os
from PIL import Image
from scipy.misc import imresize
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

WORKING_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(WORKING_PATH,'data')
TEST_PATH = os.path.join(DATA_PATH,'test')
TRAIN_PATH = os.path.join(DATA_PATH,'train')

"""
###############################################################################
REQUIRED PART
###############################################################################
"""

def load_images(n=0, train=True):
    """
    Import either test or training images into a pd.dataframe object of np.arrays (images) and image numbers
    [] -> [image_data, image_number]
    """
    print("Loading the image files...")
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

def resize_images(images):
    """
    Squeezes the image data into the average size of the image data-set. All images will be transformed into the same width and height.
    [image_data] -> [image_data]
    """
    print("Resizing the images...")
    sizes = pd.DataFrame()
    sizes['WH'] = images['image_data'].apply(lambda x: np.shape(x))
    sizes['W'] = sizes['WH'].apply(lambda x:x[0])
    sizes['H'] = sizes['WH'].apply(lambda x:x[1])
    means = sizes[['W','H']].mean()
    new_sizes = (int(round(means['W'])),int(round(means['H'])))
    images['image_data'] = images['image_data'].apply(lambda x: imresize(x,new_sizes))
    images['image_data'] = images['image_data'].apply(lambda x: minmax_scale(x).astype(np.float32))
    return images

def add_labels(images):
    """
    Add lable numbers column to the dataframe of images.
    [image_data, image_number] -> [image_data, image_number, class_id]
    """
    print("Adding the image labels...")
    labels = pd.read_csv(os.path.join(DATA_PATH,'train_onelabel.csv'))
    labels.columns = ['image_number','class_id']
    labels['image_number'] = labels['image_number'].apply(lambda x:int(x.split('.')[0]))
    required_labels = labels[labels.image_number.isin(list(images['image_number']))]
    return pd.merge(images,required_labels,on='image_number')

"""
###############################################################################
OPTIONAL PART
###############################################################################
"""

def flatten_image_data(images):
    """
    Transform the image from a n*m matrix to a 1*nm vector.
    [image_data] -> [image_data]
    """
    print('Flattening the image data...')
    images['image_data'] = images['image_data'].apply(lambda x: x.flatten())
    return images

def add_label_hotmap(images):
    """
    Convert the output label to a hotmap of zeros and ones.
    [class_id] -> [class_hotmap]
    """
    print('Creating an output hotmap...')
    n = max(images['class_id'])
    images['class_hotmap'] = images['class_id'].apply(lambda x: hotmap(x,n))
    return images

def hotmap(loc, n):
    """
    Function for turing a single row of class_id table into a hotmap
    """
    lst = np.zeros(n, dtype=np.float32)
    lst[loc-1] = 1
    return lst

def split_data(images, p = 0.3):
    """
    Split the datafram into x_train, x_test, y_train, y_test
    """
    x_data = images['image_data']
    labels = images['class_hotmap']
    x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=p)
    return x_train, x_test, y_train, y_test
