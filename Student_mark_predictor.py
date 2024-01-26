import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading the dataset
data=pd.read_csv('./Dataset/student_info.csv')

print(data.head())

print(f"We have {data.shape} (rows,columns)")
print(data.info())
print(data.describe())


plt.title('Scatter Plot of Student marks and study hours')
plt.xlabel('Student Study Hours')
plt.ylabel('Student Marks')
plt.scatter(x=data['study_hours'],y=data['student_marks'],color='red')
plt.show()

#checking for Null values
print(data.isnull())
#check the total sum of null values
print(data.isnull().sum())
#columns are missing
print(data.mean())
#fill the null values by mean
p_data=data.fillna(data.mean())
print(p_data.isnull().sum())

x=p_data[['study_hours']]
y=p_data['student_marks']

#training and testing of the model
from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3,random_state=51)
# 70% train 30% test

# Load and train the machine learning model
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred=y_pred.round(2)

#plotting my model
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,c='r')
plt.title('Plotting the prediction line into the dataset')
plt.xlabel('Study hours')
plt.ylabel('Student Marks')
plt.show()

#creating new data frame
data2=pd.DataFrame(np.c_[x_test,y_test,y_pred],columns=["study_hours","student_marks_original","student_marks_predicted"])

print(data2.head())

#Accuracy Score
from sklearn.metrics import r2_score
print(f"Acuuracy: {r2_score(y_test,y_pred).round(2)*100}%")

#saving the model to reuse
import joblib
joblib.dump(model,"Student_mark_predictor_model.pkl")
