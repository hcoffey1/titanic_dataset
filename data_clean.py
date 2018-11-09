import pandas as pd 

raw_data = pd.read_csv("train.csv")
raw_data = raw_data.drop(["PassengerId","Name", "Ticket", "Cabin"], axis=1)

raw_data["Sex"] = raw_data["Sex"].replace("male", 0)
raw_data["Sex"] = raw_data["Sex"].replace("female", 1)

raw_data["Embarked"] = raw_data["Embarked"].replace("C", 0)
raw_data["Embarked"] = raw_data["Embarked"].replace("S", 1)
raw_data["Embarked"] = raw_data["Embarked"].replace("Q", 2)


for i in raw_data: 
    raw_data[i] = pd.to_numeric(raw_data[i])
    raw_data[i] = raw_data[i].fillna((int)(raw_data[i].mean()))

raw_data.to_csv("train_clean.csv", index=None)