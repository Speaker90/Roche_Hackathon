import pandas as pd
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model file")
args = vars(ap.parse_args())

# initialize the class labels for the Kaggle dogs vs cats dataset
CLASSES = ["ASD", "No ASD"]


# load the network
print("[INFO] loading network architecture and weights...")
model = load_model(args["model"])

SHAPE = (61, 73, 61)

def load_mat_from_txt(fn):
    """ Loads data from text file, reshapes them, and
        returns a Numpy array (= degree centrality map). """

   #Note: Depending on your NumPy version you might have to remove the encoding = 'utf-8' part.
    mat = np.loadtxt(fname = fn, encoding = 'utf-8')
    mat = mat.reshape(SHAPE)
    return(mat)

#import the meta-data
data_pheno = pd.read_csv('data/train/train.csv', encoding = 'utf-8')

#get the sample size
N=len(data_pheno.index)

#initialize the feature and label vector
x=np.zeros((10,)+SHAPE)
y=np.zeros(10)

#start data import
print("[INFO] importing data...")
for i in range(N-11,N):
    x[i-N+10,] = load_mat_from_txt('data/train/' + data_pheno.fn_image_txt[i])
    y[i-N+10] = data_pheno.DX_GROUP[i]

y=y-1
y=y.astype(int)
print(y)
#normalize data to the range [0,1]
x_test=x/max(x.flatten())

# classify the image using our extracted features and pre-trained
# neural network
probs = model.predict(x_test)
#loop through the ten test pictures
for i in range(0,x.shape[0]):
    #get probabilities
    prob=probs.item(i)
    image=x[i,:,:,30]
    #get the prediction class
    prediction = int(round(prob))
    #flip probability for class 0
    if prob<0.5:
        prob=1-prob
    # draw the class and probability on the test image and display it
    # to our screen
    label = "{:.2f}%: {}\n TRUE:   {}".format(
    prob * 100,CLASSES[prediction],CLASSES[y[i]])
    fig, ax = plt.subplots()
    imgplot = ax.imshow(image)
    ax.annotate(label,(5,5),color='white')
    plt.waitforbuttonpress()
