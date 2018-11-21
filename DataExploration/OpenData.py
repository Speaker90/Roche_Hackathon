import sys
import os

import numpy as np
import pandas as pd

# %matplotlib inline  # In case you use Jupyter notebooks
import matplotlib.pyplot as plt


## Load phenotypic data
 # Summary spreadsheet that contains 
 # phenotypic data and quality assessment information
data_pheno = pd.read_csv('data/train/train.csv', encoding = 'utf-8')


## Example: Load degree centrality map 
SHAPE = (61, 73, 61)

def load_mat_from_txt(fn):
    """ Loads data from text file, reshapes them, and
        returns a Numpy array (= degree centrality map). """

    # Note: Depending on your NumPy version you might have to remove the encoding = 'utf-8' part.
    mat = np.loadtxt(fname = fn, encoding = 'utf-8')
    mat = mat.reshape(SHAPE)
    
    return(mat)
    
    
fnx = 'data/train/' + data_pheno.fn_image_txt[0]
cn = load_mat_from_txt(fnx)


## Visualize slices of degree centrality map
imgplot = plt.imshow(cn[:, :, 30])
