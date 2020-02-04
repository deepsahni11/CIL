import numpy as np
import h5py
import pickle
import sklearn
import random 
import pdb
from sklearn.metrics import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from scipy import stats
from tqdm import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from imblearn.metrics import geometric_mean_score
from numpy.random import permutation
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




seed = 0
samplers1 = [
    # Oversampling methods:
    RandomOverSampler(random_state=seed), 
    SMOTE(random_state=seed),             
    ADASYN(random_state=seed),            
    BorderlineSMOTE(random_state=seed),
    SVMSMOTE(random_state=seed),
    
    # Undersampling methods:
    RandomUnderSampler(random_state=seed),
    ClusterCentroids(random_state=seed),
    NearMiss(version=1, random_state=seed),
    NearMiss(version=2, random_state=seed)
    
    
]

samplers2 = [
    
#     NearMiss(version=3, random_state=seed),
    TomekLinks(random_state=seed),
    EditedNearestNeighbours(random_state=seed),
    RepeatedEditedNearestNeighbours(random_state=seed),
    AllKNN(random_state=seed),
    CondensedNearestNeighbour(random_state=seed),
    OneSidedSelection(random_state=seed),
    NeighbourhoodCleaningRule(random_state=seed),
    InstanceHardnessThreshold(random_state=seed),

    
    
    
    
]

samplers3 = [
# Combos:
    
    SMOTEENN(random_state=seed),
    SMOTETomek(random_state=seed)
    
]


samplers_all = [
    # Oversampling methods:
    RandomOverSampler(random_state=seed), 
    SMOTE(random_state=seed),             
    ADASYN(random_state=seed),            
    BorderlineSMOTE(random_state=seed),
    SVMSMOTE(random_state=seed),
    
    # Undersampling methods:
    RandomUnderSampler(random_state=seed),
    ClusterCentroids(random_state=seed),
    NearMiss(version=1, random_state=seed),
    NearMiss(version=2, random_state=seed),
    NearMiss(version=3, random_state=seed),
    TomekLinks(random_state=seed),
    EditedNearestNeighbours(random_state=seed),
    RepeatedEditedNearestNeighbours(random_state=seed),
    AllKNN(random_state=seed),
    CondensedNearestNeighbour(random_state=seed),
    OneSidedSelection(random_state=seed),
    NeighbourhoodCleaningRule(random_state=seed),
    InstanceHardnessThreshold(random_state=seed),
    
    
    # Combos:
    SMOTEENN(random_state=seed),
    SMOTETomek(random_state=seed)

]
samplers_array_all = np.array(samplers_all)

samplerAbbrev = [
    "ROS",
    "SMOTE",
    "ADASYN",
    "B-SMOTE",
    "SVMSMOTE",
    "RUS",
    "CC",
    "NM-1",
    "NM-2",
    "NM-3",
    "Tomek",
    "ENN",
    "RENN",
    "AkNN",
    "CNN",
    "OSS",
    "NCR",
    "IHT",
    "SMOTE+ENN",
    "SMOTE+Tomek"
]

from sklearn import metrics
def evalSamplingp(sampler, classifier, Xtrain, Xtest,ytrain, ytest):
    """Evaluate a sampling method with a given classifier and dataset
    
    Keyword arguments:
    sampler -- the sampling method to employ. None for no sampling
    classifer -- the classifier to use after sampling
    train -- (X, y) for training
    test -- (Xt, yt) for testing
    
    Returns:
    A tuple containing precision, recall, f1 score, AUC of ROC, Cohen's Kappa score, and 
    geometric mean score.
    """
    X = Xtrain
    y = ytrain
    Xt = Xtest
    yt = ytest
    
    if sampler is not None:
        X_resampled, y_resampled = sampler.fit_sample(X, y)
        classifier.fit(X_resampled, y_resampled)
    else:
        classifier.fit(X, y)
        
    yp = classifier.predict(Xt)
    yProb = classifier.predict_proba(Xt)[:,1] # Indicating class value 1 (not 0)

    precision = precision_score(yt, yp)
#     recall    = recall_score(yt, yp)
#     f1        = f1_score(yt, yp)
#     rocauc    = roc_auc_score(yt, yProb)
#     kappa     = cohen_kappa_score(yt, yp)
#     gmean     = geometric_mean_score(yt, yp)
    
    return precision

def evalSamplingr(sampler, classifier, Xtrain, Xtest,ytrain, ytest):
    """Evaluate a sampling method with a given classifier and dataset
    
    Keyword arguments:
    sampler -- the sampling method to employ. None for no sampling
    classifer -- the classifier to use after sampling
    train -- (X, y) for training
    test -- (Xt, yt) for testing
    
    Returns:
    A tuple containing precision, recall, f1 score, AUC of ROC, Cohen's Kappa score, and 
    geometric mean score.
    """
    X = Xtrain
    y = ytrain
    Xt = Xtest
    yt = ytest
    
    if sampler is not None:
        X_resampled, y_resampled = sampler.fit_sample(X, y)
        classifier.fit(X_resampled, y_resampled)
    else:
        classifier.fit(X, y)
        
    yp = classifier.predict(Xt)
    yProb = classifier.predict_proba(Xt)[:,1] # Indicating class value 1 (not 0)

#     precision = precision_score(yt, yp)
    recall    = recall_score(yt, yp)
#     f1        = f1_score(yt, yp)
#     rocauc    = roc_auc_score(yt, yProb)
#     kappa     = cohen_kappa_score(yt, yp)
#     gmean     = geometric_mean_score(yt, yp)
    
    return recall

def evalSamplingf(sampler, classifier, Xtrain, Xtest,ytrain, ytest):
    """Evaluate a sampling method with a given classifier and dataset
    
    Keyword arguments:
    sampler -- the sampling method to employ. None for no sampling
    classifer -- the classifier to use after sampling
    train -- (X, y) for training
    test -- (Xt, yt) for testing
    
    Returns:
    A tuple containing precision, recall, f1 score, AUC of ROC, Cohen's Kappa score, and 
    geometric mean score.
    """
    X = Xtrain
    y = ytrain
    Xt = Xtest
    yt = ytest
    
    if sampler is not None:
        X_resampled, y_resampled = sampler.fit_sample(X, y)
        classifier.fit(X_resampled, y_resampled)
    else:
        classifier.fit(X, y)
        
    yp = classifier.predict(Xt)
    yProb = classifier.predict_proba(Xt)[:,1] # Indicating class value 1 (not 0)

#     precision = precision_score(yt, yp)
#     recall    = recall_score(yt, yp)
    f1        = f1_score(yt, yp)
#     rocauc    = roc_auc_score(yt, yProb)
#     kappa     = cohen_kappa_score(yt, yp)
#     gmean     = geometric_mean_score(yt, yp)
    
    return  f1


def extractStandardMeasures(X, y):
    """Extracts the standard measures of the given dataset
    
    Keyword arguments:
    X -- Design matrix (feature set)
    y -- Label vector
    
    Returns:
    A tuple (N, d, c) where
    N -- no. of samples
    d -- no. of features
    C -- no. of classes
    """
    (N, d) = X.shape
    c = len(set(y))
    
    return (N, d, c)



def extractStatsMeasures(X, y):
    """Extracts the standard measures of the given dataset
    
    Keyword arguments:
    X -- Design matrix (feature set)
    y -- Label vector
    
    Returns:
    A tuple (rho, rs, I, sw, ad) where 
    rho -- Pearson correlation coefficient
    rs -- Spearman coefficient
    I -- Mutual information
    sw -- Shapiro-Wilk test for normality
    nt -- D'Agostino-Pearson test for normality
    sdr -- Non-homogeneity measure: it works well for data that is multivariate gaussian "CHECK" 
    """
    
    rho = np.corrcoef(X, rowvar=False) # Pearson Correlation Coefficient
    rs,_ = stats.spearmanr(X) # Spearman Correlation Coefficient
    
    I = [] # Mutual Information
    for c in range(X.shape[1]): # For every column
        I.append(mutual_info_regression(X, X[:,c]))
    I = np.array(I)
    
    sw = stats.shapiro(X) # Shapiro-Wilk test
    nt = stats.normaltest(X) # D'Agostino-Pearson test
    
    # Homogeneity of Class Covariance Matrices
    (N, d, c) = extractStandardMeasures(X, y)
    classList = list(set(y))
    Ni = []
    Si_inv = []
    for className in classList:
        idx = y==className
        Ni.append(sum(idx))
        Si_inv.append(np.linalg.inv(np.cov(X[idx,:], rowvar=False)))
    S = np.cov(X, rowvar=False)
    
    t1 = sum([1/(x-1) for x in Ni])
    t2 = 1/(N-c)
    gamma = 1 - (2*d*d + 3*d - 1) * (t1 - t2) / (6*(d+1)*(c-1))
    
    t3 = 0
    for i in range(c):
        t3 += (Ni[i] - 1)*np.log(np.linalg.det(np.dot(Si_inv[i], S)))
    
    M = gamma * t3
    sdr = np.exp(M / (d * sum([(x-1) for x in Ni])))
    
    return rho, rs, I, sw, nt, sdr




def extractDataSparesness(X, y):
    """Extract the data sparsity ratio of the given datasetc "CHECK" this measure beacuse rs.correlation does not work for mutivariate data
    
    Keyword arguments:
    X -- Design matrix (feature set)
    y -- Label vector
    
    Returns:
    The data sparsity ratio (N/Nm)
    """
    (N, d, c) = extractStandardMeasures(X, y)
    (rho, rs, I, sw, nt, sdr) = extractStatsMeasures(X, y)
    
    if rs.correlation <= 0.7: # for Normal and uncorrelated data 
        Nm = 2*d*c + c
    elif sdr < 1.5: # for Normal and correlated data with homogeneous covariance matrix 
        Nm = d*d + d*c + c
    else:    # for Normal and correlated data with non-homogenous covariance matrix
        Nm = c*N*N + N*c + c
    
    return N/Nm




import sys

class Graph(): 
  
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)] 
  
    # A utility function to print the constructed MST stored in parent[] 
    def printMST(self, parent=None):
        if parent is None:
            parent = self.root
        print("Edge \tWeight")
        for i in range(1,self.V): 
            print(parent[i],"-",i,"\t",self.graph[i][ parent[i] ])
  
    # A utility function to find the vertex with  
    # minimum distance value, from the set of vertices  
    # not yet included in shortest path tree 
    def minKey(self, key, mstSet): 
  
        # Initilaize min value 
        min = sys.maxsize 
  
        for v in range(self.V): 
            if key[v] < min and mstSet[v] == False: 
                min = key[v] 
                min_index = v 
  
        return min_index 
  
    # Function to construct MST for a graph  
    # represented using adjacency matrix representation 
    def primMST(self): 
  
        #Key values used to pick minimum weight edge in cut 
        key = [sys.maxsize] * self.V 
        parent = [None] * self.V # Array to store constructed MST 
        # Make key 0 so that this vertex is picked as first vertex 
        key[0] = 0 
        mstSet = [False] * self.V 
  
        parent[0] = -1 # First node is always the root of 
  
        for cout in range(self.V): 
  
            # Pick the minimum distance vertex from  
            # the set of vertices not yet processed.  
            # u is always equal to src in first iteration 
            u = self.minKey(key, mstSet) 
  
            # Put the minimum distance vertex in  
            # the shortest path tree 
            mstSet[u] = True
  
            # Update dist value of the adjacent vertices  
            # of the picked vertex only if the current  
            # distance is greater than new distance and 
            # the vertex in not in the shotest path tree 
            for v in range(self.V): 
                # graph[u][v] is non zero only for adjacent vertices of m 
                # mstSet[v] is false for vertices not yet included in MST 
                # Update the key only if graph[u][v] is smaller than key[v] 
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]: 
                        key[v] = self.graph[u][v] 
                        parent[v] = u 
        
        self.root=parent
#         self.printMST(parent) 

g = Graph(5) 
g.graph = [ [0, 2, 0, 6, 0], 
            [2, 0, 3, 8, 5], 
            [0, 3, 0, 0, 7], 
            [6, 8, 0, 0, 9], 
            [0, 5, 7, 9, 0]] 
  
g.primMST(); 


def extractDecisionBoundaryMeasures(X, y, seed=0):
    """Extract the decision boundary measures of the given dataset
    
    Keyword arguments:
    X -- Design matrix (feature set)
    y -- Label vector
    
    Returns:
    A tuple (linSep, decBoundCompVar, decBoundComp) where
    linSep -- Linear separability
    # decBoundCompVar -- Variation in decision boundary complexity # n/a for now
    decBoundComp -- Complexity of the decition boundary
    hyperCentres -- Centres of the hyperspheres along with the last column as the corresponding class
    """
    
    # Linear separability: 10-fold statified cross validation error rate using LDA-Bayes classifier 
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=seed)
    lda = LinearDiscriminantAnalysis()
    np.random.seed(seed)
    errors = []
    for trainIdx, testIdx in sss.split(X, y):
        Xs, ys = X[trainIdx], y[trainIdx] # Sample set
        Xt, yt = X[testIdx], y[testIdx] # Test set
        
        lda.fit(Xs, ys)
        pred = lda.predict(Xt)
        
        errors.append(1 - f1_score(yt, pred)) # Error = 1 - F1_score
    
    linSep = np.mean(errors)
    
    # Complexity of decision boundary: compute hyperspheres
    pool = list(range(X.shape[0]))
    size = X.shape
    np.random.shuffle(pool)
    dist = metrics.pairwise.euclidean_distances(X)
    
    hyperCentres = []
    while len(pool) > 0:
        hs = [pool[0]]          # Initialize hypersphere
        centre = X[pool[0]]     # and its centre
        hsClass = y[pool[0]]          # Class of this hypersphere 
        pool.remove(pool[0])    # Remove the initial point from the pool
        mostDistantPair = None
        
        while True and len(pool)>0:
            dist = np.sqrt(np.sum((X[pool] - centre)**2, axis=1))
            nn = pool[np.argmin(dist)]  # Nearest neighbour index
            if y[nn] != hsClass:        # If this point belongs to a different class
                break                   # conclude the set of points in this sphere
            hs.append(nn)               # Otherwise add it to the sphere
            pool.remove(nn)             # and remove it from the pool
            
#             if mostDistantPair is None: # Update the most distant pair
#                 mostDistantPair = (hs[0], hs[1])
#             else:
#                 maxDist = np.sqrt(np.sum((X[mostDistantPair[0]] - X[mostDistantPair[1]])**2)) 
#                 dist = np.sqrt(np.sum((X[hs] - X[nn])**2, axis=1))
#                 if np.max(dist) > maxDist:
#                     mostDistantPair = (hs[np.argmax(dist)], nn)
                    
#             centre = (X[mostDistantPair[0]]+X[mostDistantPair[1]])/2 # Update the centre
            centre = np.mean(X[hs], axis=0)
                
        hyperCentres.append(list(centre)+[hsClass])
    
    # Produce a MST using the centres of the hyperspheres as nodes
    hyperCentres = np.array(hyperCentres)
#     print(hyperCentres)
#     print(metrics.pairwise.euclidean_distances(hyperCentres[:,0:2]))
    
#     pdb.set_trace()
    
    g = Graph(hyperCentres.shape[0])
    g.graph = metrics.pairwise.euclidean_distances(hyperCentres[:,:2])
    g.primMST()
    # Find the number of inter-class edges in the MST
    idx1 = list(range(1,hyperCentres.shape[0]))
    idx2 = g.root[1:]
    
    N_inter = sum(hyperCentres[idx1,size[1]] != hyperCentres[idx2,size[1]])
    decBoundComp = N_inter / hyperCentres.shape[0] 
    
    return (linSep, decBoundComp, N_inter,hyperCentres)



from scipy.special import gamma

def extractTopologyMeasures(X, y):
    """Extract the topology measures of the given dataset
    
    Keyword arguments:
    X -- Design matrix (feature set)
    y -- Label vector
    
    Returns:
    A tuple (groupsPerClass, sdVar, scaleVar) where
    samplesPerGroup
    groupsPerClass
    sdVar -- Variation in feature standard deviation
    # scaleVar -- Scale variation [Not yet implemented]
    """
    
    (linSep, decBoundComp, _,hyperCentres) = extractDecisionBoundaryMeasures(X, y)
    
    samplesPerGroup = hyperCentres.shape[0] / X.shape[0]
    classes = list(set(hyperCentres[:,hyperCentres.shape[1]-1]))
    groupsPerClass = [sum(hyperCentres[:,hyperCentres.shape[1]-1]==c) for c in classes]
    sdVar = [np.std(np.std(X[y==c], axis=1)) for c in classes]

    return (samplesPerGroup, groupsPerClass, sdVar)





# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.neural_network import MLPClassifier



#### Every dataset has been labelled using this sequence 
#### (flip_fraction,num_informative,class_separation,num_clusters,random_seed,num_features,num_classes,
####  num_repeated,num_redundant)

num_datapoints = 1000

flip_fraction = [0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
num_informative = [3,4,5]
class_separation = np.arange(0.30, 2.0, 0.1).tolist()
num_clusters = [1,2,3]

random_seed = 0 
num_features = 5
num_classes = 2
num_repeated = 0
num_redundant = 0
weights = [[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.6,0.4],[0.5,0.5]]



data = []
data2 = []





X_train_datasets_5d = []
y_train_datasets_5d = []
X_test_datasets_5d = []
y_test_datasets_5d = []


c = 0

for w in weights:
    for f in flip_fraction:
        for num_i in num_informative:
            for cs in class_separation:
                for num_c in num_clusters:


                    c = c+1
                    X,y = make_classification(n_samples=num_datapoints, n_features=num_features, n_informative=num_i, 
                                        n_redundant=num_redundant, n_repeated=num_repeated, n_classes=num_classes, n_clusters_per_class=num_c,
                                           class_sep=cs,
                                       flip_y=f,weights=w, random_state = random_seed)



                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
                    sss.get_n_splits(X, y)
                    for train_index, test_index in sss.split(X, y):
                        Xtrain, Xtest = X[train_index], X[test_index]
                        ytrain, ytest = y[train_index], y[test_index]


                        X_train_datasets_5d.append(Xtrain)
                        y_train_datasets_5d.append(ytrain)
                        X_test_datasets_5d.append(Xtest)
                        y_test_datasets_5d.append(ytest)

len(X_train_datasets_5d)





import timeit

start = timeit.default_timer()


data = []
data2 = []
data3 = []

for j in range(len(X_train_datasets_5d)):
# for j in range(1):
    
    data.append(tuple(["Dataset-" + str(j+1),"","","","","","","","","","","","","","","","","","","",""]))
    Xtrain = X_train_datasets_5d[j]
    ytrain = y_train_datasets_5d[j]
    Xtest = X_test_datasets_5d[j]
    ytest = y_test_datasets_5d[j]
    row = ["Precision"]
    for sampler in samplers_array_all:
        t = ""
    #                         precision, recall, f1, rocauc, kappa, gmean = evalSampling(sampler, RandomForestClassifier(max_depth=2, random_state=0), Xtrain, Xtest, ytrain, ytest)
    #                         print(precision)
        try:
            precision = evalSamplingp(sampler, MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 10), batch_size = 1,max_iter = 200, random_state=1), Xtrain, Xtest, ytrain, ytest)
    #                             print(" &", round(precision,3), end="")
            t = str(round(precision,3))
        except:
    #                             print(" &", "N/A", end="")
            t = "N/A"

        row.append(t)


    data.append(tuple(row))

    rowdash = ["Recall"]
    for sampler in samplers_array_all:
        t = ""
        try:
            recall = evalSamplingr(sampler,  MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 10), batch_size = 1,max_iter = 200, random_state=1), Xtrain, Xtest, ytrain, ytest)
    #                             print(" &", round(recall,3), end="")
            t = str(round(recall,3))
        except:
    #                             print(" &", "N/A", end="")
            t = "N/A"

        rowdash.append(t)

    data2.append(tuple(rowdash))
    

    row3 = ["F_1"]
    for sampler in samplers_array_all:
        t = ""
        try:
            
            f1 = evalSamplingf(sampler,  MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 10), batch_size = 1,max_iter = 200, random_state=1), Xtrain, Xtest, ytrain, ytest)
#                             print(" &", round(f1,3), end="")
            t = str(round(f1,3))
        except:
            
#                             print(" &", "N/A", end="")
            t = "N/A"

        row3.append(t)

    data3.append(tuple(row3))
    
    print("Dataset " + str(j) + "done!!")




stop = timeit.default_timer()

print('Time: ', stop - start) 

# np.savetxt('E:\Internships_19\Internship(Summer_19)\Imbalanced_class_classification\Class_Imabalanced_Learning_Code\CIL Code\RESULTS\Data_metrics_8415_datasets_nn_precision.csv', data, delimiter=',', fmt=['%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s'  ], header = "Metrics,ROS,SMO,ADA,B-S,S-S,RUS,CC,NM1,NM2,NM3,Tomek,ENN,RENN,AkNN,CNN,OSS,NCR,IHT,SMOTE+ENN,SMOTE+Tomek",comments='')
# np.savetxt('E:\Internships_19\Internship(Summer_19)\Imbalanced_class_classification\Class_Imabalanced_Learning_Code\CIL Code\RESULTS\Data_metrics_8415_datasets_nn_recall.csv', data2, delimiter=',', fmt=['%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s'  ], header = "Metrics,ROS,SMO,ADA,B-S,S-S,RUS,CC,NM1,NM2,NM3,Tomek,ENN,RENN,AkNN,CNN,OSS,NCR,IHT,SMOTE+ENN,SMOTE+Tomek",comments='')
# np.savetxt('E:\Internships_19\Internship(Summer_19)\Imbalanced_class_classification\Class_Imabalanced_Learning_Code\CIL Code\RESULTS\Data_metrics_8415_datasets_nn_f1.csv', data3, delimiter=',', fmt=['%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s'  ], header = "Metrics,ROS,SMO,ADA,B-S,S-S,RUS,CC,NM1,NM2,NM3,Tomek,ENN,RENN,AkNN,CNN,OSS,NCR,IHT,SMOTE+ENN,SMOTE+Tomek",comments='')
