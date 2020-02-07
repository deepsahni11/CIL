import numpy as np
import sklearn
import random 
import pdb
from sklearn.metrics import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit

prediction_y = load('../Data_metrics_8415_datasets_y_prediction.npy')
y_test_datasets_5d_resampled = load('../Data_metrics_8415_datasets_y_test.npy')


data = []
data2 = []
data3 = []


for i in range(len(y_test_datasets_5d)):
    
    row = ["Dataset" + str(i+1)]
    rowdash = ["Dataset" + str(i+1)]
    row3 = ["Dataset" + str(i+1)]
    
    for j in range(21):
        
        print(i*21+j)
        
        print(np.shape(prediction_y[i*21+j]))
        print(np.shape(y_test_datasets_5d_resampled[i*21+j]))
        ypred = prediction_y[i*21+j]
        ytest = y_test_datasets_5d_resampled[i*21+j].squeeze(1)
        
        print(ytest)
        print(ypred)
        
        t = ""
   
        try:
            precision = evalSamplingp(ytest, ypred)
            t = str(round(precision,3))
        except:
            t = "N/A"

        row.append(t)
        
        
        t = ""
        try:
            recall = evalSamplingr(ytest, ypred)
        except:
            t = "N/A"

        rowdash.append(t)
        
        
        t = ""
        try:
            
            f1 = evalSamplingf(ytest, ypred)
            t = str(round(f1,3))
        except:
            
            t = "N/A"

        row3.append(t)


    data.append(tuple(row))
    data2.append(tuple(rowdash))
    data3.append(tuple(row3))
    
    print("Dataset " + str(i) + "done!!")



np.savetxt('../Data_metrics_8415_datasets_nn_precision.csv', data, delimiter=',', fmt=['%s' ,'%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s'  ], header = "Metrics,No_sampler,ROS,SMO,ADA,B-S,S-S,RUS,CC,NM1,NM2,NM3,Tomek,ENN,RENN,AkNN,CNN,OSS,NCR,IHT,SMOTE+ENN,SMOTE+Tomek",comments='')
np.savetxt('../Data_metrics_8415_datasets_nn_recall.csv', data2, delimiter=',', fmt=['%s' ,'%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s'  ], header = "Metrics,No_sampler,ROS,SMO,ADA,B-S,S-S,RUS,CC,NM1,NM2,NM3,Tomek,ENN,RENN,AkNN,CNN,OSS,NCR,IHT,SMOTE+ENN,SMOTE+Tomek",comments='')
np.savetxt('../Data_metrics_8415_datasets_nn_f1.csv', data3, delimiter=',', fmt=['%s' ,'%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s' , '%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s' ,'%s','%s'  ], header = "Metrics,No_sampler,ROS,SMO,ADA,B-S,S-S,RUS,CC,NM1,NM2,NM3,Tomek,ENN,RENN,AkNN,CNN,OSS,NCR,IHT,SMOTE+ENN,SMOTE+Tomek",comments='')
