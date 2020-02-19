from sklearn.neural_network import MLPRegressor
import numpy as np
import sklearn
import random 
import pdb
import pandas as pd



data_x = np.array(pd.read_csv("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\RESULTS\\162_datasets\\Good_data_metrics_162_datasets_6_features.csv") )
# data_x = data_x[:,1:]
data_y = np.array(pd.read_excel("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\RESULTS\\162_datasets\\Dataset_metrics_162_datasets_nn_final_precision_no_spaces.xlsx")) 



training_x = data_x
# [:140][2].reshape(-1, 1)
training_y = data_y
# [:140].max(axis = 1)
# if(training_y == "N/A"):
#     training_y = 1

test_x = data_x[140:]
# [2].reshape(-1, 1)
test_y = data_y[140:]

rs_test = []
rs_train = []
training_loss = []
test_loss = []



# c = 0
# ,15,20,15,30,35,40
# ,15,20,15,30,35
for p in [15,20,25,30,35,40]:
    for q in [15,20,25,30,35,40]:
        reg = MLPRegressor(alpha=1e-5,
                           hidden_layer_sizes=(p, q),
                           random_state=1,
                           activation="tanh",
                           max_iter=300)
        predictions = []
        
        for i in range(20):
            reg.fit(training_x, training_y[:,i])

    # .reshape(-1, 1)
            pred_y_test = reg.predict(test_x)
            predictions.append(pred_y_test)


        predictions = np.transpose(predictions)


        sample_abbreviation = ["ROS","SMO","ADA","B-S","S-S","RUS","CC","NM1","NM2","NM3","Tomek","ENN","RENN","AkNN","CNN","OSS","NCR","IHT","SMOTE+ENN","SMOTE+Tomek"]
        K_list = [1,2,3,4,5,6]

        for K in K_list:
            best_fraction = []
            worst_fraction = []
            for i in range(len(predictions)):

                best_pred = np.argpartition(predictions[i,:],-K)[-K:]
                best_real = np.argpartition(test_y[i,:],-K)[-K:]


                counter = 0
                for i in range(len(best_pred)):
                    for j in range(len(best_pred)):
                        if(best_pred[i] == best_real[j]):
                            counter = counter + 1

                best_fraction.append(counter/K)


                worst_pred = np.argpartition(predictions[i,:],K)[:K]
                worst_real = np.argpartition(test_y[i,:],K)[:K]


                counter = 0
                for i in range(len(worst_pred)):
                    for j in range(len(worst_pred)):
                        if(worst_pred[i] == worst_real[j]):
                            counter = counter + 1

                worst_fraction.append(counter/K)

            print("######################## HIDDEN LAYERS (" + str(p)+"," + str(q) +  ")##########################")
            print("Fraction of the best " + str(K) + " performers: " + str(np.mean(best_fraction)))
            print("Fraction of the worst " + str(K) + " performers: " + str(np.mean(worst_fraction)))
            
        print("#########################################################################################################")
        print("#########################################################################################################")
            

