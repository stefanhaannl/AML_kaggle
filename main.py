"""
Main working directory
"""
import preprocessing
import numpy as np
import pandas as pd

def load_data(n=0,train=True):
    df = preprocessing.load_images(n,train)
    if train == True:
        df = preprocessing.add_labels(df)
    return df