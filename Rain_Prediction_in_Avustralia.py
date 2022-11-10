import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv')
df.head()

#Data Preprocessing
#-----------------------------------
#Converting Categorical Variables to binay variables via get_dummies()
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

df_sydney_processed.replace(["No","Yes"], [0,1], inplace = True)

#Training Data and Test Data
df_sydney_processed.drop("Date",axis = 1, inplace = True)
df_sydney_processed = df_sydney_processed.astype(float)

#X Values(Features) and Y(Target Value)
X = df_sydney_processed.drop(columns="RainTomorrow", axis = 1)
Y = df_sydney_processed["RainTomorrow"]

#LinearRegression

#Train_Test_Split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 10)

#LinReg
LinearReg = LinearRegression()

#LinearReg Fit
LinearReg.fit(x_train,y_train)

#Predict x_test
predictions = LinearReg.predict(x_test)

#MAE,MSE,R2 via predictions and y_test
from sklearn.metrics import r2_score
LinearRegression_MAE = np.mean(np.absolute(predictions - y_test))
LinearRegression_MSE = np.mean((predictions, y_test))
LinearRegression_R2 = r2_score(y_test,predictions)

#MAE, MSE, R2 in a tabular format using data frame for the linear model
data = np.array([[LinearRegression_MAE,LinearRegression_MSE,LinearRegression_R2]])
reports = pd.DataFrame(data = data, columns = ["MAE","MSE","R2"])
reports

#Create and Train KNN model, Training data(x_train,y_train), n_neighbors = 4
neigh = 4
KNN = KNeighborsClassifier(n_neighbors = neigh).fit(x_train,y_train)
KNN

#Predict testing data(x_test), save to array
predictions = KNN.predict(x_test)
predictions

#Predcit, y_test dataframe calculate the value for each metric
KNN_Accuracy_Score = metrics.accuracy_score(y_test,predictions)
KNN_JaccardIndex = metrics.jaccard_score(y_test,predictions)
KNN_F1_Score = metrics.f1_score(y_test,predictions)

#Reports
data_KNN = np.array([[KNN_Accuracy_Score,KNN_JaccardIndex,KNN_F1_Score]])
reports_KNN= pd.DataFrame(data = data_KNN, columns = ["Accuracy_Score","JaccardIndex","F1_Score"])
reports_KNN

#Decision Tree(Using the training Data)
Tree = DecisionTreeClassifier().fit(x_train,y_train)
Tree

#Predict xtest for Tree
predictions = Tree.predict(x_test)
predictions

#Tree Prediction
Tree_Accuracy_Score = metrics.accuracy_score(y_test,predictions)
Tree_JaccardIndex = metrics.jaccard_score(y_test,predictions)
Tree_F1_Score = metrics.f1_score(y_test,predictions)

#Reports Tree
data_Tree = np.array([[Tree_Accuracy_Score,Tree_JaccardIndex,Tree_F1_Score]])
reports_Tree= pd.DataFrame(data = data_Tree, columns = ["Accuracy_Score","JaccardIndex","F1_Score"])
reports_Tree

#Logistic Regression

#Train_test_split, test_size = 0.2, random_state = 1
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2, random_state = 1)

#Create and Train LogisticRegression Model data (x_train,y_train) solver = liblinear
LR = LogisticRegression(solver = "liblinear").fit(x_train,y_train)

#Predict x_test, save it to the array predictions
predictions = LR.predict(x_test)
predictions

#Predictions scores y_test
LR_Accuracy_Score = metrics.accuracy_score(y_test,predictions)
LR_JaccardIndex = metrics.jaccard_score(y_test,predictions)
LR_F1_Score = metrics.f1_score(y_test,predictions)
LR_Log_Loss = metrics.log_loss(y_test,predictions)

#Reports LR
data_LR = np.array([[LR_Accuracy_Score,LR_JaccardIndex,LR_F1_Score,LR_Log_Loss]])
reports_LR= pd.DataFrame(data = data_LR, columns = ["Accuracy_Score","JaccardIndex","F1_Score","Log_Loss"])
reports_LR

#SVM using training data
SVM = svm.SVC(kernel = "linear").fit(x_train,y_train)
SVM

#Predict x_test, save it to the array predictions
predictions = SVM.predict(x_test)
predictions

#Predictions scores y_test, 
SVM_Accuracy_Score = metrics.accuracy_score(y_test,predictions)
SVM_JaccardIndex = metrics.jaccard_score(y_test,predictions)
SVM_F1_Score = metrics.f1_score(y_test,predictions)

#Reports SVM
data_SVM = np.array([[SVM_Accuracy_Score,SVM_JaccardIndex,SVM_F1_Score]])
reports_SVM= pd.DataFrame(data = data_SVM, columns = ["Accuracy_Score","JaccardIndex","F1_Score"])
reports_SVM

#All of the Reports
data_all = np.array([[LinearRegression_MAE,LinearRegression_MSE,LinearRegression_R2,float("NaN")],[KNN_Accuracy_Score,KNN_JaccardIndex,KNN_F1_Score,float("NaN")],[Tree_Accuracy_Score,Tree_JaccardIndex,Tree_F1_Score,float("NaN")],[LR_Accuracy_Score,LR_JaccardIndex,LR_F1_Score,LR_Log_Loss],[SVM_Accuracy_Score,SVM_JaccardIndex,SVM_F1_Score,float("NaN")]])
reports_all = pd.DataFrame(data = data_all , columns = ["Accuracy_Score","JaccardIndex","F1Score","Log_Loss"])
reports_all