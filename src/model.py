import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
import joblib
from imblearn.over_sampling import SMOTE
from smart_open import open
from sklearn.neighbors import NearestNeighbors


def run(fold):

    df = pd.read_csv("folded_data.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    xtrain = df_train.drop("Crop_Damage",axis=1).values
    ytrain = df_train.Crop_Damage.values

    xvalid = df_valid.drop("Crop_Damage",axis=1).values
    yvalid = df_valid.Crop_Damage.values

    sm = SMOTE(sampling_strategy = "all",k_neighbors=2)
    xtrain, ytrain = sm.fit_resample(xtrain, ytrain)



    model = xgb.XGBClassifier(objective="multi:softmax",n_jobs=-1)

    model.fit(xtrain,ytrain)

    preds = model.predict(xvalid)

    acc = metrics.accuracy_score(yvalid,preds)

    print(f"fold {fold} ACC {acc}")

    print(metrics.classification_report(preds,yvalid))
    
    joblib.dump(model,f"model/model_{fold}.bin")


if __name__  =="__main__":

    run(0)
    run(1)
    run(2)
    run(3)
    run(4)