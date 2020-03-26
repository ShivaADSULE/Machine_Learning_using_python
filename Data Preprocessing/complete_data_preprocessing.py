import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# get data
data = pd.read_csv("Study Data\Data.csv")

# Removing Missing Values 
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Salary"] = data["Salary"].fillna(data["Salary"].mean())


# Handling Catagorical data and seperating input and output
required_out = pd.get_dummies(data["Purchased"],drop_first=True).values

data = pd.get_dummies(data.iloc[:,0:3],drop_first=True).values

print(required_out)
print(data)

# Training and Tesing Spliting


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,required_out, test_size=0.25, random_state=0)

print(X_train,X_test,y_test,y_train,sep="\n")

"""
For SVM or NN
use Standard Scalling


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X.train = std.fit_transform(X.train)
X_test = std.transform(X_test)

"""