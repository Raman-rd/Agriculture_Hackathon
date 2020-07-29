import pandas as pd
import numpy as np

def clean_data(df):

    df.drop("ID",axis=1,inplace=True)

    df["Number_Weeks_Used"]=df["Number_Weeks_Used"].ffill()

    df["Estimated_Insects_Count"] = np.where(df["Estimated_Insects_Count"]>3500,3500,df["Estimated_Insects_Count"])

    df["Crop_Type"]=np.where(df["Crop_Type"]==0,1,2)

    df["Soil_Type"]=np.where(df["Soil_Type"]==0,1,2)

    # bins=13

    # df["Number_Doses_Week"]=pd.cut(df["Number_Doses_Week"],bins=bins,labels=False)

    # df["Number_Weeks_Used"]=pd.cut(df["Number_Weeks_Used"],bins=bins,labels=False)

    # df["Number_Weeks_Quit"]=pd.cut(df["Number_Weeks_Quit"],bins=bins,labels=False)

    df.to_csv("input/Cleaned_data_.csv",index=False)


if __name__ == "__main__":

    df = pd.read_csv("input/train_yaOffsB.csv")
    clean_data(df)