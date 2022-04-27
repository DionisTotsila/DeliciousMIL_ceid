import numpy as np
import pandas as pd

def dat_to_df(name):
    assert name == "train" or name == "test"
    # csv is used because rows are seperated by new lines we do not make use of the "," delimiter
    df = pd.read_csv("../data/" + name + "-data.dat", names=["data"])
    df["label"] = pd.read_csv("../data/"+ name + "-label.dat", header=None)
    return df

