#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report,plot_confusion_matrix

#Loading dataset for diabetes
df_diabetes=pd.read_csv("diabetes.csv")
df_diabetes.head()

df_diabetes['Label'] = df_diabetes['Outcome'].map({0:'Non-Diabetic', 1:'Diabeteic'})
#correlation 
df_corr=df_diabetes.corr()
plt.figure(figsize=(20,8))
sns.heatmap(df_corr, annot=True, cmap='Blues')
plt.show()

#Training our model
features=df_diabetes[['Glucose','BloodPressure', 'DiabetesPedigreeFunction']]
x=features
y=df_diabetes['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30, random_state=8)

# Creating a Support Vector Machine model.
m1= SVC(kernel = 'poly', gamma= 'scale')
m1.fit(x_train,y_train)

pred_train=m1.predict(x_train)
pred_test=m1.predict(x_test)

score_train=metrics.accuracy_score(y_train,pred_train)
score_test=metrics.accuracy_score(y_test,pred_test)


print(plot_confusion_matrix(m1,x_test,y_test, values_format='d'))
print(classification_report(y_test,pred_test))

#Streamlit part
@st.cache()
def prediction(Glucose,BloodPressure, DiabetesPedigreeFunction):
  label = m1.predict([[Glucose,BloodPressure, DiabetesPedigreeFunction]])
  label = label[0]
  if label == 0:
    return "The person does not have Diabetes"
  else:
    return "The person is Diabetic"

st.sidebar.title("Diabetes Prediction App")

glucose_level = st.sidebar.slider("Enter Glucose value", 0, 119)
bloodp_level = st.sidebar.slider("Enter BloodPressure", 0, 122)
dpf = st.sidebar.slider("Enter DiabetesPedigreeFunction", 0.078, 2.42)

if st.sidebar.button("Predict"):
  classified=prediction(glucose_level, bloodp_level, dpf)
  st.sidebar.write(classified)
  st.sidebar.write("The accuracy of the model is : ", score_test)

#introducing graphs

st.sidebar.title("Visualisation Selector")
st.sidebar.subheader("For Diabetes Prediction")
drop_list=st.sidebar.multiselect("Select the Charts/Plots",("Countplot", "Piechart"))

st.set_option('deprecation.showPyplotGlobalUse', False)

if "Countplot" in drop_list:
  st.subheader("Countplot")
  plt.figure(figsize=(20,8))
  sns.countplot(x="Outcome", data=df_diabetes)
  st.pyplot()
if "Piechart" in drop_list:
  st.subheader("Piechart")
  pie_data=df_diabetes['Outcome'].value_counts()
  plt.figure(dpi=108)
  plt.pie(pie_data, labels=pie_data.index, autopct="%1.3f%%")
  st.pyplot()

#dataset for heart disease
df_heart_dis=pd.read_csv("heart.csv")
df_heart_dis.head()

df_diabetes['Label'] = df_diabetes['Outcome'].map({0:'Person do not have a heart disease', 1:'Person have a heart disease'})

df_corr_heart=df_heart_dis.corr()
plt.figure(figsize=(20,8))
sns.heatmap(df_corr_heart, annot=True)
plt.show()

#Training our model
features=df_heart_dis[['cp','thalach', 'slope']]
x=features
y=df_heart_dis['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30, random_state=42)

m2= SVC(kernel = 'linear')
m2.fit(x_train,y_train)

pred_train=m2.predict(x_train)
pred_test=m2.predict(x_test)

score_train=metrics.accuracy_score(y_train,pred_train)
score_test2=metrics.accuracy_score(y_test,pred_test)

print(plot_confusion_matrix(m2,x_test,y_test, values_format='d'))
print(classification_report(y_test,pred_test))

@st.cache()
def prediction(cp,thalach, slope):
  label2 = m2.predict([[cp,thalach, slope]])
  label2 = label2[0]
  if label2 == 0:
    return "The person does not have a heart disease"
  else:
    return "The person have a heart disease"

st.title("Heart Disease Prediction App")  

chestpaint= st.radio("Enter Chest Pain Type",(1,2,3,4))
maxheartr = st.slider("Enter Maximum Heart Rate", 90, 162)
slopepeak = st.number_input("Enter the slope of peak exercise", 0, 2)

if st.button("Predict Heart Disease"):
  classified2= prediction(chestpaint, maxheartr, slopepeak)
  st.write(classified2)
  st.write("Accuracy score of this model is:", score_test2)

#st.sidebar.title("Visualisation Selector")
#st.sidebar.subheader("For Heart Disease Prediction")

#st.sidebar.subheader("Scatterplot")
#feat_list=st.sidebar.multiselect("Select values for X-axis", ('cp', 'thalach', 'dpf'))
#st.set_option('deprecation.showPyplotGlobalUse', False)
#for i in feat_list:
#  st.subheader("Scatterplot for "+str(i))
#  plt.figure(figsize=(20,8))
#  sns.scatterplot(x=i, y ="target", data=df_heart_dis)
# st.pyplot()
