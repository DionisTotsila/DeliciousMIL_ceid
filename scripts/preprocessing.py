import numpy as np
import pandas as pd
import re 
from sklearn.preprocessing import MinMaxScaler

# read .dat dataset and create pandas dataframe
def dat_to_df(name):
    assert name == "train" or name == "test"
    # csv is used because rows are seperated by new lines we do not make use of the "," delimiter
    df = pd.read_csv("data/" + name + "-data.dat", names=["data"])
    df["label"] = pd.read_csv("data/"+ name + "-label.dat", header=None)
    return df

# create Bag of Words vector from dataframe row 
def bow(row):
    text  = row['data']
    # create BoW dict
    # zeros intialize use range as keys
    diction = dict.fromkeys((str(i) for i in range(8520)), 0)
    
    #remove <num> substrings from text
    #print(text)
    text = re.sub('<[0-9]+>',"",text)
    #print(text)
    # fix double spaces occuring
    text = ' '.join(text.split())
    
    #iterate through every number
    for numbers in re.findall(re.compile('[0-9]+'), text):
        
        diction[numbers] = diction[numbers] + 1
    return (np.array(list(diction.values())))


test_df = dat_to_df("test")
train_df = dat_to_df("train")

test_df['bow'] = test_df.apply(bow, axis = 1)
train_df['bow'] = train_df.apply(bow, axis = 1)

print(test_df)
print(train_df)


# Feature scaling

def scale(df):
    temp = df['bow'].values
    temp = np.array(temp.tolist())
    # init scaler
    scaler = MinMaxScaler().fit(temp)
    # scale data
    X_scaled = scaler.transform(temp)
    # create row with scaled data
    df['scaled'] = pd.DataFrame(((x,) for x in X_scaled), columns = ['scaled'])
    return df


test_df = scale(test_df)
train_df = scale(train_df)

# save  new datasets
test_df.to_csv("data/test_df.csv")
train_df.to_csv("data/train_df.csv")

