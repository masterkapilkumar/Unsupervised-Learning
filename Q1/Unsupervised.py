import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from timeit import default_timer as timer

def load_train_data(train_folder):
    data = []
    for file in os.listdir(train_folder):
        data.append(np.load(os.path.join(train_folder,file)))
    return np.array(data)

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
    
if __name__=='__main__':
    
    start = timer()
    
    if(len(sys.argv)==3):
        train_data = load_train_data(sys.argv[1])
        test_data = load_test_data(sys.argv[2])
    else:
        print("Using default arguments")
        train_data = load_train_data("train/")
        test_data = load_test_data("test.npy")
    
    print("Reading data complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    