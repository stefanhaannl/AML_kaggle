"""
Preprocessing file for loading the information in a workable format.
"""
import numpy as np
import pandas as pd
import os

WORKING_PATH = os.path.abspath()
DATA_PATH = os.path.join(WORKING_PATH,'data')

def load_images(n=0, train=True):
    """
    Import either test or training images into a pd.dataframe object of np.arrays (images) and label numbers
    """
    pass

def add_labels(images, train=True):
    """
    Transform pd.series object of np.arrays (images) into pd.series of images and text labels
    """
    pass
    