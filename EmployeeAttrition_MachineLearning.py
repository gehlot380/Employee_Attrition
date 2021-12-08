#Step1 : importing all the libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns #needed for heatmap
import numpy as np


#Step 2: loading the data
hr_data = pd.read_csv('HR-Employee-Attrition.csv') 
print(hr_data.head())

#To get the number of rows and number of columns in the data
#print(hr_data.shape)

#Step 3: Cleaning and checking the missing values
print(hr_data.isnull().any()) #returns false meaning no null values

#Step 4 : Transfroming non-numeric data to numeric data
from sklearn.preprocessing import LabelEncoder

for column in hr_data.columns:
   if hr_data[column].dtype == np.number: 
       #if column's data type is a number, we don't want to do anything
       #and just continue
    continue
   hr_data[column] = LabelEncoder().fit_transform(hr_data[column]) #encode the column

#Step 5: Data analysis
#heat map to see the correlation of independent and dependent variables
plt.figure(figsize=(18, 18))
sns.heatmap(hr_data.corr(),annot =True, fmt = '.0%')

#Plot to see attrition w.r.t years at company
fig_dims = (10, 4)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(x='YearsAtCompany', hue='Attrition', data = hr_data, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1));

#scatter plot to see monthly income, age, and attrition
sns.lmplot(x = 'Age', y = 'MonthlyIncome', data = hr_data, col = 'Attrition')
plt.show()

#Attrition and distance from home
sns.barplot(y='DistanceFromHome', x = 'Attrition', data = hr_data)
plt.show()

#Attrition and number of companies worked
sns.boxplot(x='Attrition', y='NumCompaniesWorked', data=hr_data)
plt.show()


#Create a new column at the end of the dataframe that contains the same value 
hr_data['Age_Years'] = hr_data['Age']
#Remove the first column called age 
hr_data = hr_data.drop('Age', axis = 1)

#Step 6 : Modeling Process - Random Forest
#Matrix of independent variable is created first followed by a dependent variable.
X = hr_data.iloc[:, 1:32].values #independent variables 
#all rows selected, and all the columns till last are selected
#iloc function helps in slicing the data frame,helps in selecting a piece from the dataset 
Y = hr_data.iloc[:, 0].values #dependent variables

# importing train_test_split() 
from sklearn.model_selection import train_test_split
#The size of the split is specified via test size
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state = 20)
#The random state will be used as a seed to the random number generator to ensures that the
#random numbers are generated in the same order

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, random_state = 20)
#number of estimators, random state is defined

model.fit(X_train, Y_train) #data needs to be trained on training dataset

prediction_model = model.predict(X_test) #to make predictions
#print(prediction_model)

#Step 7 : Model Validation - Random Forest
#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,prediction_model))

#to check the accuracy of the prediction model on test data
from sklearn import metrics # metrics are easy to compare
print("Accuracy = ", metrics.accuracy_score(Y_test,prediction_model))

from sklearn.metrics import classification_report
print(classification_report(Y_test,prediction_model))


#Step 8 :Model process - Naive Bayes 
#Pre-processing of data is already performed
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()

NB.fit(X_train, Y_train) #data needs to be trained on training dataset

prediction_model = NB.predict(X_test) #to make predictions
print(prediction_model)



#Step9: Model Validation -  Naive Bayes
#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,prediction_model))

#to check the accuracy of the prediction model on test data
from sklearn import metrics # metrics are easy to compare
print("Accuracy = ", metrics.accuracy_score(Y_test,prediction_model))

    