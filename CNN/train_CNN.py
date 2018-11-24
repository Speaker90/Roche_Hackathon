import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

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
N=len(data_pheno.index)-10

#initialize the feature and label vector
x=np.zeros((N,)+SHAPE)
y=np.zeros(N)

#start data import
print("[INFO] importing data...")
for i in range(0,N):
    x[i,] = load_mat_from_txt('data/train/' + data_pheno.fn_image_txt[i])
    y[i] = data_pheno.DX_GROUP[i]
    if i > 0 and i % 25 == 0:
	print("[INFO] processed {}/{}".format(i,N))

#normalize data to the range [0,1]
x=x/max(x.flatten())
y=y-1

#create the test sample
(x_train, x_test, y_train, y_test) = train_test_split(
	x, y, test_size=0.25, random_state=42)

#define the model   
model = Sequential()

model.add(Conv2D(488, (4, 4), activation='relu', input_shape=SHAPE))
model.add(Conv2D(488, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(976, (3, 3), activation='relu'))
model.add(Conv2D(976, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(732, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(366, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


model.fit(x_train, y_train,
          epochs=200,
          batch_size=24)
# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(x_test, y_test,
	batch_size=24, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))

model.save('output/new_CNN.hdf5')
