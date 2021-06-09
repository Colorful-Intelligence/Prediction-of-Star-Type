"""
Prediction of Star Types
"""

#%% Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

from sklearn.metrics import accuracy_score

#%% Read the Dataset

data = pd.read_csv("star_dataset.csv")

#%% Exploratory Data Analysis (EDA)

data.shape # (240, 7)

data.info()

data.describe()

data.columns
"""
['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)',
       'Absolute magnitude(Mv)', 'Star type', 'Star color', 'Spectral Class']
"""

# Target column is "Star type" , I'll change the name of this column  as a "Target"

data.rename({"Star type":"Target"},axis = 1,inplace = True)


"""
float64 = Luminosity(L/Lo) , Radius(R/Ro) , Absolute magnitude(Mv)
int64 = Temperature (K)
object = Star color , Spectral Class
"""

#%% Correlation Matrix

corr_matrix = data.corr()
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(corr_matrix,annot = True,linewidths=0.5,fmt =".2f" ,ax = ax)
plt.title("Correlation Matrix Between Features (Columns)")
plt.show()

#%% Numerical Variables = Temperature (K),Luminosity(L/Lo),Radius(R/Ro) , Absolute magnitude(Mv) 

def plot_hist(numerical_variable):
    plt.figure(figsize = (10,10))
    plt.hist(data[numerical_variable],bins = 50)
    plt.xlabel(numerical_variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(numerical_variable))
    plt.show()

numerical_variables = ["Temperature (K)","Luminosity(L/Lo)","Radius(R/Ro)","Absolute magnitude(Mv)"]
for N in numerical_variables:
    plot_hist(N)


#%% Categorical Variables = Target, Star color, Spectral Class

def bar_plot(variable):
    
    # get value
    var = data[variable]
    
    # count number of the variable
    varValue = var.value_counts()
    
    # visualize
    
    plt.figure(figsize = (10,10))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()

categorical_variables = ["Target", "Star color", "Spectral Class"]

for C in categorical_variables:
    bar_plot(C)
    
#%% Missing Values
data.columns[data.isnull().any()]
data.isnull().sum()   
# Dataset hasn't any missing value.

#%% Outlier Detection

def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        # 1 st quartile
        Q1 = np.percentile(df[c],25)
        
        # 3 rd quartile
        Q3 = np.percentile(df[c],75)
        
        # IQR
        IQR = Q3 - Q1
        
        # Outlier step
        outlier_step = IQR * 1.5
   
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers

data.loc[detect_outliers(data,['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)',
       'Absolute magnitude(Mv)'])]

#%% Label Encoder Operation

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

columnsToEncode = list(data.select_dtypes(include=["object"]))

for feature in columnsToEncode:
    data[feature] = le.fit_transform(data[feature])
    

#%% To get X and Y Coordinates

y = data.Target.values
x_data = data.drop(["Target"],axis = 1)

#%% Normalization Operation

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

#%% Train-Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#%% K-Nearst Neighbors Classification

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
knn_predicted = knn.predict(x_test)

print("Accuracy of the KNN for k = 3 : % {}".format(accuracy_score(y_test,knn_predicted)*100))

"""
Accuracy of the KNN for k = 3 : % 93.75
"""

# Can we improve accuracy of the KNN model , Let's do it

score_list = []

for each in range(1,100):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    knn2_predicted = knn2.predict(x_test)
    score_list.append(accuracy_score(y_test,knn2_predicted))

plt.plot(range(1,100),score_list)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy")
plt.show()

# Graph shows that best k values are 1,2 and 3 

#%% Naive Bayes Classification

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)
nb_predicted = nb.predict(x_test)
print("Accuracy of the Naive Bayes Classification : % {} ".format(accuracy_score(y_test,nb_predicted)*100))

"""
Accuracy of the Naive Bayes Classification : % 95.83333333333334 
"""

# Accuracy of the model has  incaresed 

# Can we increase of the model ? 

#%% Random Forest Classification

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,random_state=1)
rf.fit(x_train,y_train)
rf_predicted = rf.predict(x_test)

print("Accuracy of the Random Forest Classification : % {}".format(accuracy_score(y_test, rf_predicted)*100))

"""
Accuracy of the Random Forest Classification : % 100.0
"""

#%% Confusion Matrix

y_pred = rf_predicted
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

# Confusion Matrix Visualize 
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor = "red",fmt = ".0f",ax = ax)
plt.title("Confusion Matrix")
plt.show()



#%% K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = rf, X = x_train,y = y_train,cv = 10)

print("average accuracy = % {}".format(np.mean(accuracies)*100))
print("average std = % {} ".format(np.std(accuracies)*100))

"""
average accuracy = % 99.47368421052632
average std = % 1.578947368421054 
"""



