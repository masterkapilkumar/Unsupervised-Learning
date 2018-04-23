import numpy as np
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import random

from timeit import default_timer as timer

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

def Q2(train_x, train_y, test_data, model=None):
    pca = PCA(n_components=50)
    x = train_x.reshape((train_x.shape[0]*train_x.shape[1],train_x.shape[2]))
    
# =============================================================================
#     #make validation data set
#     A_t = np.array(random.sample(range(100000),5000))
#     valid_x = x[A_t]
#     valid_y = train_y[A_t]
# =============================================================================
    
    x_scaled = StandardScaler().fit_transform(x)
    x_projected = pca.fit_transform(x_scaled)
    
    pca = PCA(n_components=50)
    test_scaled = StandardScaler().fit_transform(test_data)
    test_projected = pca.fit_transform(test_scaled)
        
    
    print("PCA done...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    if(model):
        print("ye to hona hi tha!")
        predictions = predict_from_model(model, x_projected)
#        predictions = np.array(list(map(lambda a:labels_map[a], predictions)))
        np.savetxt("predictions.csv", predictions, delimiter=",")
        correct = 0
        for (corr,pred) in zip(train_y, predictions):
            if(corr==pred): correct += 1
        accuracy = correct/x.shape[0] * 100
        print("Training Accuracy: %.2f%%" %(accuracy) )
        
        predictions = predict_from_model(model, test_projected)
        predictions = np.array(list(map(lambda a:labels_map[a], predictions)))
        np.savetxt("test_predictions.csv", predictions, delimiter=',')
        return
    
    #make svm classifier
    clf = svm.SVC(decision_function_shape='ovo', verbose=True, C=10, gamma=0.05)
    
# =============================================================================
#     #tune parameters
#     Cs = [0.1, 1, 5, 10, 15]
#     gammas = [0.01, 0.05, 0.5, 1, 2, 5]
#     param_grid = {'C': Cs, 'gamma' : gammas}
#     clf = GridSearchCV(clf, param_grid, cv=10)
#     clf.fit(x_projected, valid_y)
#     pickle.dump(clf, open("svm_model", 'wb'))
#     print(clf.best_params_)
#     #optimal parameters found: C=5, gamma=0.01
# =============================================================================
    
    #train svm model using optimal parameters
    clf.fit(x_projected, train_y)
    pickle.dump(clf, open("svm_model", 'wb'))
    
    print("Libsvm training complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    predictions = clf.predict(x_projected)
#    predictions = np.array(list(map(lambda a:labels_map[a], predictions)))
    np.save("train_predictions", predictions)
    correct = 0
    for (corr,pred) in zip(train_y, predictions):
        if(corr==pred): correct += 1
    accuracy = correct/x.shape[0] * 100
    print("Training accuracy: %.2f%%" %(accuracy) )
    
    print("Libsvm train set prediction complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
	
    predictions = clf.predict(test_projected)
    predictions = np.array(list(map(lambda a:labels_map[a], predictions)))
    np.savetxt("test_predictions.csv", predictions, delimiter=',')
    
    print("Libsvm testing complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
#    np.savetxt("libsvm_in.csv", x_projected, delimiter=",")
#    
#    print("Writing to file done...")
#    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
def predict_from_model(model_file, data):
    clf = pickle.load(open(model_file, 'rb'))
    return clf.predict(data)
    
if __name__=='__main__':
    
    start = timer()
    
    if(len(sys.argv)==3):
        train_x, train_y, labels_map = load_train_data(sys.argv[1])
        test_data = load_test_data(sys.argv[2])
    else:
        print("Using default arguments")
        train_x, train_y, labels_map = load_train_data("train/")
        test_data = load_test_data("test.npy")
    
    print("Reading data complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    #visualize_vector(test_data[1])
    Q2(train_x, train_y, test_data, "svm_model")             #TODO:vapply PCA then scale data
    