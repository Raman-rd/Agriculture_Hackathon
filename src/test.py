import joblib
import pandas as pd
import numpy as np

model = joblib.load("model/model_4.bin")

df = pd.read_csv("input/Cleaned_test_data_.csv")

df["fold"] =-1
 

X = df.values


p = model.predict(X)

pred_df = pd.DataFrame(p)


pred_df.to_csv("preds.csv")