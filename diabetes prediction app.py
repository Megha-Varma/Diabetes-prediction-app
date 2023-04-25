#Back end
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sns


diabetes_df = pd.read_csv('C:/Users/niles/OneDrive/Documents/Megha/clevered internship programme/Diabetes prediction app/Dataset/diabetes (2) - Copy - Copy.csv')
diabetes_df.head()

diabetes_df.columns

diabetes_df.info()

diabetes_df.describe()

diabetes_df.describe().T

diabetes_df.isnull().head(10)

diabetes_df.isnull().sum()

p = diabetes_df.hist(figsize = (20,20))

color_wheel = {1: "#0392cf", 2: "#7bc043"}
colors = diabetes_df["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_df.Outcome.value_counts())
p=diabetes_df.Outcome.value_counts().plot(kind="bar")


X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,
                                                    random_state=7)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_train = rfc.predict(X_train)
from sklearn import metrics

print("Accuracy_Score =", format(metrics.accuracy_score(y_train, rfc_train)))

from sklearn import metrics

predictions = rfc.predict(X_test)
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, predictions)))

Accuracy_Score = 0.7677165354330708

print(X_train)
print(X_test)
print(y_train)
print(y_test)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

from sklearn import metrics

predictions = dtree.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test,predictions)))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))

from xgboost import XGBClassifier

xgb_model = XGBClassifier(gamma=0)
xgb_model.fit(X_train, y_train)

from sklearn import metrics

xgb_pred = xgb_model.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test, xgb_pred)))

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)

svc_pred = svc_model.predict(X_test)

from sklearn import metrics

print("Accuracy Score =", format(metrics.accuracy_score(y_test, svc_pred)))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test,svc_pred))

rfc.feature_importances_

(pd.Series(rfc.feature_importances_, index=X.columns).plot(kind='barh'))

import pickle

saved_model = pickle.dumps(rfc)

rfc_from_pickle = pickle.loads(saved_model)

rfc_from_pickle.predict(X_test)

diabetes_df.head()

diabetes_df.tail()





#front end

import tkinter as tk

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 700, height = 500, bg='skyblue')

canvas1.pack()

#header
label1 = tk.Label(root, text=' DIABETES PREDICTION', font = "Times 30 bold",bg= 'aquamarine2')
canvas1.create_window(350,40, window=label1)

#1st variable-pregnancies
label2 = tk.Label(root, text=' NUMBER OF PREGNANCIES: ', font = 'bold')
canvas1.create_window(175, 120, window=label2)

entry1 = tk.Entry (root, font = 'bold', width = 15) 
canvas1.create_window(420, 120, window=entry1)

#2nd variable-BMI
label3 = tk.Label(root, text=' BMI: ', font = 'bold')
canvas1.create_window(175, 200, window=label3)


entry2 = tk.Entry (root, font = 'bold', width = 15)
canvas1.create_window(420, 200, window=entry2)

#3rd variable-variable-age
label4 = tk.Label(root, text=' AGE: ', font = 'bold')
canvas1.create_window(175, 280, window=label4)


entry3 = tk.Entry (root, font = 'bold', width = 15)
canvas1.create_window(420, 280, window=entry3)

#prediction using backend
def values():
  PREGNANCIES = float(entry1.get())
 
  BMI = float(entry2.get())
  
  AGE = float(entry3.get())
 
  Prediction_result = ( rfc.predict([[PREGNANCIES , BMI,  AGE ]]))   
  
  if list(Prediction_result) == [0]:
     open_Toplevel1() 
  else:
     open_Toplevel2()  




#prediction button
button1 = tk.Button (root, text='DETERMINE RESULT',command=values, bg='lightgreen', font = "Times 20 bold") 
canvas1.create_window(350, 400, window=button1)




#info1

def info1():
    
    global hidden
    if hidden:
        pregnancy_info.pack(side = "top", fill = tk.BOTH)
        hidden = False
    else:
        pregnancy_info.pack_forget()
        hidden=True
        
hidden = True

pregnancy_info = tk.Label(root, text= 'Mention the number \n of pregnancies \n incurred during the \n lifetime.Write (0) if \n not applicable',bg='LightGoldenrod1', width = 20, height = 10, borderwidth=3, relief="solid")




info_btn1 = tk.Button (root, text=' ? ',command=info1, bg='khaki') 
canvas1.create_window(530, 120, window=info_btn1)

#info2

def info2():
    
    global hidden
    if hidden:
        BMI_info.pack(side = "top", fill = tk.BOTH)
        hidden = False
    else:
        BMI_info.pack_forget()
        hidden=True
        
hidden = True

BMI_info = tk.Label(root, text= 'calculating your Body Mass Index (BMI) :\n  it is your weight in kilograms divided by the square of your height in meters.',bg='LightGoldenrod1', width = 20, height = 10, borderwidth=3, relief="solid")




info_btn2 = tk.Button (root, text=' ? ',command=info2, bg='khaki') 
canvas1.create_window(530, 200, window=info_btn2)

#info3

def info3():
    
    global hidden
    if hidden:
        age_info.pack(side = "top", fill = tk.BOTH)
        hidden = False
    else:
        age_info.pack_forget()
        hidden=True
        
hidden = True

age_info = tk.Label(root, text= 'Enter your current age.',bg='LightGoldenrod1', width = 20, height = 10, borderwidth=3, relief="solid")


info_btn3 = tk.Button (root, text=' ? ',command=info3, bg='khaki') 
canvas1.create_window(530, 280, window=info_btn3)

#refresh button
def clear_text():
  entry1.delete(0, END) 
                      
    
  entry2.delete(0, END)
  
  entry3.delete(0, END)

END='end'

clear_btn = tk.Button (root, text=' REFRESH ',command=clear_text, bg='khaki', font='bold') 
canvas1.create_window(650,480, window=clear_btn)
        

root.mainloop()


from PIL import ImageTk, Image

def open_Toplevel1(): 
     
    pass

    top1 = tk.Toplevel(root)
     
    
    top1.title("Not Diabetic")
     
    canvas2 = tk.Canvas(top1, width = 700, height = 500, bg='skyblue')

    canvas2.pack()


    
    label1 = tk.Label(top1, text = "CONGRATULATIONS\n DIABETES TEST \n RESULTED NEGATIVE\n   \n PRECAUTINARY STEPS \n ENCOURAGED", font ='bold', bg= 'lightgreen', borderwidth = 3, relief = "solid")
    canvas2.create_window(525,250, window=label1)  
    
    button1 = tk.Button(top1, text = "Exit",command = top1.destroy)
    canvas2.create_window(680,10, window=button1) 
    

    img = ImageTk.PhotoImage(Image.open("C:/Users/niles/OneDrive/Documents/Megha/clevered internship programme/Diabetes prediction app/images/resized.jpg"))   

    label2 = tk.Label(top1, width=345, height=495, image = img)
    canvas2.create_window(180,250, window=label2)  
    
     
    
    top1.mainloop()
 


from PIL import ImageTk, Image

def open_Toplevel2(): 
     
    pass

    top2 = tk.Toplevel(root)
     
    
    top2.title("Diabetic")
     
    canvas3 = tk.Canvas(top2, width = 700, height = 500, bg='skyblue')

    canvas3.pack()


    
    label1 = tk.Label(top2, text = "DIABETES TEST \n RESULTED POSITIVE\n   \n TAKE NECESSARY  \n STEPS AND VISIT A \n DOCTOR ", font ='bold', bg= 'CORAL1', borderwidth = 3, relief = "solid")
    canvas3.create_window(525,250, window=label1)  
    
    button1 = tk.Button(top2, text = "Exit",command = top2.destroy)
    canvas3.create_window(680,10, window=button1) 
    

    img = ImageTk.PhotoImage(Image.open("C:/Users/niles/OneDrive/Documents/Megha/clevered internship programme/Diabetes prediction app/images/Diabetes-Infographic RESIZED.jpg"))   

    label2 = tk.Label(top2, width=345, height=495, image = img)
    canvas3.create_window(180,250, window=label2)  
    
     
    
    top2.mainloop()

