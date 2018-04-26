import numpy as np
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle

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

def save_submission_file(data, pred_file):
    f = open(pred_file,'w')
    f.write("ID,CATEGORY\n")
    for i,pred in enumerate(data):
        f.write(str(i)+","+str(pred)+'\n')
    f.close()

def Q1(train_x, train_y, test_data, model=None, pred_file="predictions.txt"):
    x = train_x.reshape((train_x.shape[0]*train_x.shape[1],train_x.shape[2]))
    if(not model):
        ##########TRAINING##############
        kmeans = KMeans(n_clusters=20, n_init=10)
        kmeans.fit(x)
        pickle.dump(kmeans, open("kmeans_model", 'wb'))
        model = kmeans
        
        print("Kmeans clustering complete...")
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    else:
        model = pickle.load(open(model, 'rb'))
    ###############TESTING################
    
    local_labels_map = defaultdict(str)
    #train set prediction
    labels = model.labels_
    total_correct = 0
    for i in range(20):
        examples_labels = train_y[np.where(labels==i)]
        counts = np.bincount(examples_labels)
        c = np.argmax(counts)
        local_labels_map[i] = c
        total_correct += counts[c]
    
    accuracy = total_correct/labels.shape[0] * 100
    print("Training Accuracy: %.2f%%" %(accuracy) )
    print("Kmeans train set prediction complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    
    #test set prediction
    #predictions = model.predict(test_data)
    predictions = []
    centroids = model.cluster_centers_
    for ex in test_data:
        min_dist = 1000000000
        min_cluster = 0
        for c in range(20):
            dist = np.linalg.norm(ex-centroids[c])
            if(dist<min_dist):
                min_dist = dist
                min_cluster = c
        predictions.append(labels_map[local_labels_map[min_cluster]])
    
    save_submission_file(predictions, pred_file)
    
    print("Kmeans test set prediction complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    
def Q2(train_x, train_y, test_data, model=None, pred_file="predictions.txt"):
    x = train_x.reshape((train_x.shape[0]*train_x.shape[1],train_x.shape[2]))
    
# =============================================================================
#     #make validation data set
#     A_t = np.array(random.sample(range(100000),5000))
#     valid_x = x[A_t]
#     valid_y = train_y[A_t]
# =============================================================================
    
    pca = PCA(n_components=50)
    x_projected = pca.fit_transform(x)/255.0
    test_projected = pca.fit_transform(test_data)/255.0
    
    print("PCA done...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    #if the model does not exist then first train
    if(not model):
        ##########TRAINING##############
        #make svm classifier
        clf = svm.SVC(decision_function_shape='ovo', verbose=True, shrinking=False)
        
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
        
        model = clf
        
        print("Libsvm training complete...")
        print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    else:
        model = pickle.load(open(model, 'rb'))
    ###############TESTING################
    #train set prediction
    predictions = predict_from_model(model, x_projected)
    correct = 0
    for (corr,pred) in zip(train_y, predictions):
        if(corr==pred): correct += 1
    accuracy = correct/x.shape[0] * 100
    print("Training Accuracy: %.2f%%" %(accuracy) )
    print("Libsvm train set prediction complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
    #test set prediction
    predictions = predict_from_model(model, test_projected)
    predictions = np.array(list(map(lambda a:labels_map[a], predictions)))
    save_submission_file(predictions, pred_file)
    
    print("Libsvm test set prediction complete...")
    print("Time taken: %.2fs\n"%(calculate_time_elapsed()))
    
def predict_from_model(model, data):
    return model.predict(data)
    
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
    
    aa=Q1(train_x, train_y, test_data, "kmeans_model")
    
    #visualize_vector(test_data[1])
    #Q2(train_x, train_y, test_data)             #TODO: apply PCA then scale data
    