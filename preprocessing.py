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
        path = "/train/"
    else:
        path = "/test/"
    
    for filename in os.listdir(DATA_PATH+path):
        x = np.array([np.array(Image.open(DATA_PATH+path+filename))])
        im_list.append(x) 
        id_im = int(filename.split('.')[0])
        label_list.append(id_im)
        n-=1
        if n == 0:
            break
                
    data_dict = {"image":im_list,"number":label_list}
    df = pd.DataFrame(data_dict)
    print df
        
        
    pass

def add_labels(images, train=True):
    """
    Transform pd.dataframe object of np.arrays (images) into pd.series of images and text labels
    """
    pass
    