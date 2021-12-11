import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('insurance.csv')

num_variable = (df.dtypes==float) | (df.dtypes=="int64")
num_variable = df.columns[num_variable].tolist()

cat_variable = df.dtypes==object
cat_variable = df.columns[cat_variable].tolist()
df["age_range"] = 1000
for i in range(len(df["age"])):
    if df["age"][i]<30:
        df["age_range"][i] = 1
    elif df["age"][i] >=30 and df["age"][i]<45:
        df["age_range"][i] = 2
    elif df["age"][i] >=45:
        df["age_range"][i] = 3

df["have_children"] = ["No" if i == 0 else "Yes" for i in df["children"]]
cat_variable.append("have_children")
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df[cat_variable] = df[cat_variable].apply(lambda col: lb.fit_transform(col.astype(str)))
X = df.drop(columns=["charges","region"])
y = df["charges"]
print(X.columns)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train,y_train)

forest_score = forest.score(X_test,y_test)
print(forest_score)
pickle.dump(forest, open("model_weights.pkl", "wb"))

