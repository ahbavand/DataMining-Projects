import pandas as pd
import numpy as np
from sklearn import tree

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


test_one = test

test_one["Survived"] = 0



train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"]=train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Age"] = train["Age"].fillna(train["Age"].median())


# Impute the missing value with the median
test.Fare[152] = test.Fare.median()
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1


test["Embarked"]=test["Embarked"].fillna("S")
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test["Age"] = test["Age"].fillna(train["Age"].median())






features = train[["Pclass", "Sex", "Age", "Fare","SibSp", "Parch", "Embarked"]].values
test_features = test[["Pclass", "Sex", "Age", "Fare","SibSp", "Parch", "Embarked"]].values

target = train["Survived"].values
max_depth = 10
min_samples_split =5
my_tree=tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,random_state=1)
my_tree.fit(features,target)
my_prediction = my_tree.predict(test_features)
print(my_prediction)




PassengerId =(test["PassengerId"]).astype(int)
my_solution_sex = pd.DataFrame(my_prediction,PassengerId,columns=["Survived"])
my_solution_sex.to_csv("my_solution1_sex.csv", index_label = ["PassengerId"])

