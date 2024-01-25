import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#import clean dataset 
data = pd.read_csv('clean_data.csv')

#Data transformation
#using standard scalar to transform data to a consistent scale across all variables

def scaling_data(data, variables):
    
    for var in variables:
        #convert data to a numpy array and reshape
        new_data = np.array(data[var])
        new_data = new_data.reshape(-1,1)
            
        #fit and transform data
        scaled_data = StandardScaler().fit_transform(new_data)
        
        #update values in dataframe
        data[var]=scaled_data.flatten()

    return data


#variables to scale
#all of these variables are significantly different between diabetics and non-diabetics
vars_to_scale = list(data.columns)
#remove outcomes column
vars_to_scale.pop()

#scaling data
scaled_data = scaling_data(data, vars_to_scale)

#PREDICTION MODELS
#x and y variables
model_x = scaled_data.iloc[:, :-1]
model_y = scaled_data.iloc[:, -1]

#logistic regression
log_reg = LogisticRegression(random_state=123)

#cross validation
#fits model a set number of times
#training sets are splot into smaller sets (k-1 folds), then the rest of the data is usd as the testing set
#each fold is used as the testing set once
#average score after computing through the loop is used

#using cross validation on logistic regression model
log_reg_model = cross_validate(log_reg, model_x, model_y, scoring=["roc_auc"], cv=10)

#predicted values
log_reg_y_predict = cross_val_predict(log_reg, model_x, model_y, cv=10)


#RESULTS
#AUC function
def find_auc(model_used):
    mean_auc = (model_used["test_roc_auc"]).mean()
    return mean_auc

#sensitivity and specificity function
def sensitivity_specificity(true_y, predicted_y):
    #extracting false positives, false negatives, true positive, true negative
    #use confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_y, predicted_y).ravel()
    
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    
    return sensitivity, specificity

#AUC
lr_auc = find_auc(log_reg_model)
#average auc: 0.855

#sensitivity and specificity
lr_sensitivity, lr_specificity = sensitivity_specificity(model_y , log_reg_y_predict)
#sensitivity: 0.577
#specificity: 0.881


#SVM
svc = SVC(random_state=123)

#using cv on SVM model
svc_model = cross_validate(svc, model_x, model_y, scoring=["roc_auc"], cv=10)

#predicted y values
svc_y_predict = cross_val_predict(svc, model_x, model_y, cv=10)

svc_auc = find_auc(svc_model)
#average auc: 0.842
svc_sensitivity, svc_specificity = sensitivity_specificity(model_y, svc_y_predict)
#sensitivity : 0.584
#specificity = 0.878


#decision tree
dt = DecisionTreeClassifier(random_state = 123)
dt_model = cross_validate(dt, model_x, model_y, scoring=["roc_auc"], cv=10)

#predicted y values
dt_y_predict = cross_val_predict(dt, model_x, model_y, cv=10)

dt_auc = find_auc(dt_model)
#average auc:0.685

dt_sensitivity, dt_specificity = sensitivity_specificity(model_y, dt_y_predict)
#sensitivity: 0.592
#specificity: 0.778


#random forest
rf = RandomForestClassifier(random_state=123)
rf_model = cross_validate(rf, model_x, model_y, scoring=["roc_auc"], cv=10)

#predicted y values
rf_y_predict = cross_val_predict(dt, model_x, model_y, cv=10)

rf_auc = find_auc(rf_model)
#average auc: 0.849

rf_sensitivity, rf_specificity = sensitivity_specificity(model_y, rf_y_predict)
#sensitivity: 0.592
#specificity: 0.778

#Takeaways:
#logistic regression has the highest AUC with 0.855 and specificity with 0.881
#sensitivity is 0.577, which is similar to the other models
#Therefore, the logistic regression model performed the best.

























