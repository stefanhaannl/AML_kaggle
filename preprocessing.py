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
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from skimage.transform import resize

WORKING_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(WORKING_PATH,'data')
TEST_PATH = os.path.join(DATA_PATH,'test')
TRAIN_PATH = os.path.join(DATA_PATH,'train')
AUGMENTATION_PATH = os.path.join(DATA_PATH,'augmentations')

"""
###############################################################################
REQUIRED PART
###############################################################################
"""

def load_images(n=0, train=True, size=(64,64)):
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
        im = Image.open(os.path.join(path,filename))
        im.thumbnail(size, Image.ANTIALIAS)
        x = np.array([np.array(im)])
        gray = x[0,:,:] 
        old_size = gray.shape
        if (size[0]-old_size[0]) % 2 == 0:
            pad1 = int((size[0]-old_size[0])/2)
            pad2 = pad1
        else:
            pad1 = int(np.floor((size[0]-old_size[0])/2))
            pad2 = pad1+1
        if (size[1]-old_size[1]) % 2 == 0:
            pad3 = int((size[1]-old_size[1])/2)
            pad4 = pad3
        else:
            pad3 = int(np.floor((size[1]-old_size[1])/2))
            pad4 = pad3+1
        newimg = np.lib.pad(gray,((pad1,pad2),(pad3,pad4)),mode='constant',constant_values = ((255,255),(255,255)))
        im_list.append(newimg.astype(np.float32))
        id_im = int(filename.split('.')[0])
        label_list.append(id_im)
        n-=1
        if n == 0:
            break
    if train == False:
        template = pd.read_csv(os.path.join(DATA_PATH, 'sample.csv'))
    data_dict = {"image_data":im_list,"image_number":label_list}
    df = pd.DataFrame(data_dict)
    df['image_data'] = df['image_data'].apply(lambda x: x/x.max())
    if train == True:
        return df,1 
    else:
        return df, template
    
def resize_images(images):
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
    n = max(images['class_id'])+1
    images['class_hotmap'] = images['class_id'].apply(lambda x: hotmap(x,n))
    return images

def hotmap(loc, n):
    """
    Function for turing a single row of class_id table into a hotmap
    """
    lst = np.zeros(n, dtype=np.float32)
    lst[loc] = 1
    return lst

def split_data(images, p = 0.3):
    """
    Split the datafram into x_train, x_test, y_train, y_test
    """
    x_data = images['image_data']
    labels = images['class_hotmap']
    x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=p)
    return x_train, x_test, y_train, y_test

"""
###############################################################################
KERAS
###############################################################################
"""

def get_augment(x_train, y_train, max_loops, batch_size):
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    datagen = ImageDataGenerator(rotation_range =  180, horizontal_flip= True, vertical_flip = True, fill_mode='nearest')
    datagen.fit(x_train)
    cur_loop = 0
    output_x = x_train
    output_y = y_train
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
    #reate a grid of 3x3 images
        output_x = np.concatenate((output_x,x_batch))
        output_y = np.concatenate((output_y,y_batch))
        cur_loop += 1
        if cur_loop >= max_loops:
            break
    return output_x ,output_y
            
        