import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


# load data frame 
df=pd.read_pickle("../datasets/sample.pkl")
df =df[0:50]


#print("Dataframe loaded",df.head())
""" 
# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

"""

#----------------------------------------------------------------------------

class CustomEMAKIDataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self):
        # load data
        self.df=pd.read_pickle("../datasets/sample.pkl")
        # extract labels
        self.df_labels=df['task']
        # drop non numeric columns to make it simpler for testing now 
        # conver to torch dtypes
        self.dataset=torch.tensor(df.drop(['task','type','mu'], axis = 1).values.astype(np.float32))

        self.labels=torch.tensor(self.df_labels.to_numpy().reshape(-1)).to(torch.float32)
    
    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx],self.labels[idx]

