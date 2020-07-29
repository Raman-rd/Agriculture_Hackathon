import pandas as pd
import numpy as np
from sklearn import model_selection

def create_fold(df):

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)

    y = df.Crop_Damage.values

    for f,(t,v) in enumerate(kf.split(X=df,y=y)):
        df.loc[v,"kfold"] = f

    df.to_csv("folded_data.csv",index=False)

if __name__=="__main__":

    df = pd.read_csv("input/Cleaned_data_.csv")

    create_fold(df)