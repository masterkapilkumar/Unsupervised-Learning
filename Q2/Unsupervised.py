from timeit import default_timer as timer
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import svm

from sklearn.cluster import KMeans

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation,ZeroPadding2D
from keras.optimizers import SGD

#from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils import np_utils


def load_train_data(train_folder):
    data = []
    data_y = []
    labels_map = defaultdict(str)
    i=0
    for file in os.listdir(train_folder):
        x =  np.load(os.path.join(train_folder,file))
        data.append(x)
        data_y += [i]*x.shape[0]
        labels_map[i] = os.path.splitext(file)[0]
        i+=1
    return (np.array(data, dtype='float64'), np.array(data_y), labels_map)

def load_test_data(test_file):
    return np.load(test_file)
    
def visualize_vector(vect):
    vect.resize((28,28))
    plt.imshow(vect, 'gray')
    plt.show()

def calculate_time_elapsed():
    global start
    time_elapsed = timer()-start
    start = timer()
    return time_elapsed

def Q4(train_x, train_y, test_data, valid_x=None, valid_y=None, model=None, pred_file="predictions.txt"):
    
    train_x = scale(train_x).reshape(train_x.shape[0],28,28,1)
    valid_x = scale(valid_x).reshape(valid_x.shape[0],28,28,1)
    test_data = scale(test_data).reshape(test_data.shape[0],28,28,1)
    
    model = Sequential()
    ## 1 CNN layer
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28,28,1)))
    ## Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    #model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    ## Max Pooling
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
    
    ## 1 fully connected layer
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(20, activation='softmax'))
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    train_labels_one_hot = np_utils.to_categorical(train_y)
    valid_labels_one_hot = np_utils.to_categorical(valid_y)
    model.fit(train_x, train_labels_one_hot, batch_size=100, epochs=15, verbose=1, validation_data=(valid_x, valid_labels_one_hot))
    model.save('compete_model.h5')
#    model.fit(train_x, train_labels_one_hot, batch_size=100, epochs=10, verbose=1)
    predictions = model.predict(test_data, verbose=1)
    predictions = np.argmax(predictions, axis=1)
    predictions = np.array(list(map(lambda a:labels_map[a], predictions)))
    save_submission_file(predictions, pred_file)

def make_validation_set(data_x, data_y, n, rand_file):
    indices = np.load(rand_file)
    train_x = np.array([data_x[indices[i]] for i in range(data_x.shape[0]-n)])
    train_y = data_y[indices[:-n]]
    valid_x = np.array([data_x[indices[i]] for i in range(data_x.shape[0]-n,data_x.shape[0])])
    valid_y = data_y[indices[-n:]]
    return train_x, train_y, valid_x, valid_y

def predict_from_keras_model(model_file, test_data, pred_file="predictions.txt"):
    test_data = scale(test_data).reshape(test_data.shape[0],28,28,1)
    model = load_model(model_file)
    predictions = model.predict(test_data, verbose=1)
    predictions = np.argmax(predictions, axis=1)
    predictions = np.array(list(map(lambda a:labels_map[a], predictions)))
    save_submission_file(predictions, pred_file)

def predict_from_model(model, data):
    return model.predict(data)

def save_submission_file(data, pred_file):
    f = open(pred_file,'w')
    f.write("ID,CATEGORY\n")
    for i,pred in enumerate(data):
        f.write(str(i)+","+str(pred)+'\n')
    f.close()
    
if __name__=='__main__':
    
    start = timer()
    
    if(len(sys.argv)==3):
        train_x, train_y, labels_map = load_train_data(sys.argv[1])
        test_data = load_test_data(sys.argv[2])
    else:
        print("Using default arguments")
        train_x, train_y, labels_map = load_train_data("train/")
        test_data = load_test_data("test.npy")
    
    
    
    train_x = train_x.reshape((train_x.shape[0]*train_x.shape[1],train_x.shape[2]))
    valid_size = 10000
    train_x, train_y, valid_x, valid_y = make_validation_set(train_x, train_y, valid_size, "rand_indices.npy")
    
    print("Reading data complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    Q4(train_x, train_y, test_data, valid_x, valid_y)
    #predict_from_keras_model("compete_model.h5", test_data)
    