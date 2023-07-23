import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, plot_model

np.random.seed(2)
db=pd.read_pickle("../datasets/sample.pkl")
df_gen = pd.read_csv("../datasets/gender.csv")
gender = {}
for row in df_gen.iterrows():
   gender[row[0]] = row[1]
   
mouse_data = db[db['type'].isin(['mousemove', 'mousedown', 'mouseup'])]
keyboard_data = db[db['type'].isin(['keyup', 'keydown'])]
keyboard_data=keyboard_data.drop(['ID', 'INVALID', 'X', 'Y', 'resolutionX', 'resolutionY', 'mu'], axis=1)
mouse_data=mouse_data.drop(['ID', 'INVALID', 'value', 'mu'], axis=1)
mouse_data['type'].replace('mousemove',0,inplace=True)
mouse_data['type'].replace('mousedown',1,inplace=True)
mouse_data['type'].replace('mouseup',2,inplace=True)

keystrokes = keyboard_data['value'].unique()
for i in range(len(keystrokes)):
    keyboard_data['value'].replace(keystrokes[i], i, inplace=True)

# change the type columns to 0 and 1
keyboard_data['type'].replace('keydown', 0, inplace=True)
keyboard_data['type'].replace('keyup', 1, inplace=True)
#########
db['type'].replace('mousemove',0,inplace=True)
db['type'].replace('mousedown',1,inplace=True)
db['type'].replace('mouseup',2,inplace=True)
db['type'].replace('keydown', 3, inplace=True)
db['type'].replace('keyup', 4, inplace=True)
for i in range(len(keystrokes)):
    db['value'].replace(keystrokes[i], i, inplace=True)
db['value'].replace(np.nan,inplace=True)
db['X'].replace(np.nan,0,inplace=True)
db['Y'].replace(np.nan,0,inplace=True)

random_seed = 2
def load_mouse_data():
    USERS = set(mouse_data['user'])
    X_train = pd.DataFrame(columns = ['type','X','Y'])
    X_test  = pd.DataFrame(columns = ['type','X','Y'])
    y_train = pd.DataFrame()
    y_test  = pd.DataFrame()
    for index,user in enumerate(USERS) :
     X_user= mouse_data[mouse_data['user']==user]
     X = X_user[['type','X','Y']]
     y = X_user['task']
     X_train_user, X_test_user, y_train_user, y_test_user = train_test_split(X, y, test_size=0.2, random_state=random_seed)
     X_train = pd.concat([X_train,X_train_user])
     X_test= pd.concat([X_test,X_test_user])
     y_train = pd.concat([y_train,y_train_user])
     y_test= pd.concat([y_test,y_test_user])
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    X_train = X_train.astype('float32')
    X_test=X_test.astype('float32')
    y_train=y_train.astype('float32')
    y_test = y_test.astype('float32')
    

    return X_train, X_test,y_train, y_test

def load_keyboard_data():
    USERS = set(keyboard_data['user'])
    X_train = pd.DataFrame(columns = ['value'])
    X_test  = pd.DataFrame(columns = ['value'])
    y_train = pd.DataFrame()
    y_test  = pd.DataFrame()
    for index,user in enumerate(USERS) :
     X_user= keyboard_data[keyboard_data['user']==user]
     X = X_user[['type','value']]
     y = X_user['task']
     X_train_user, X_test_user, y_train_user, y_test_user = train_test_split(X, y, test_size=0.2, random_state=random_seed)
     X_train = pd.concat([X_train,X_train_user])
     X_test= pd.concat([X_test,X_test_user])
     y_train = pd.concat([y_train,y_train_user])
     y_test= pd.concat([y_test,y_test_user])
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    X_train = X_train.astype('float32')
    X_test=X_test.astype('float32')
    y_train=y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    return  X_train, X_test,y_train, y_test


def load_combined_dataset():
    USERS = set(db['user'])
    X_train = pd.DataFrame(columns = ['type','X','Y','value'])
    X_test  = pd.DataFrame(columns = ['type','X','Y','value'])
    y_train = pd.DataFrame()
    y_test  = pd.DataFrame()
    for index,user in enumerate(USERS) :
     X_user= db[db['user']==user]
     X = X_user[['type','X','Y','value']]
     y = [gender[user-1].gender]*len(X)
     y = pd.DataFrame(y, columns=['gender'])
     print(y)
     X_train_user, X_test_user, y_train_user, y_test_user = train_test_split(X, y, test_size=0.2, random_state=random_seed)
     X_train = pd.concat([X_train,X_train_user])
     X_test= pd.concat([X_test,X_test_user])
     y_train = pd.concat([y_train,y_train_user])
     y_test= pd.concat([y_test,y_test_user])
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    X_train = X_train.astype('float32')
    X_test=X_test.astype('float32')
    y_train=y_train.astype('float32')
    y_test = y_test.astype('float32')
    
    return  X_train, X_test,y_train, y_test




def create_sliding_windows(data, targets, window_size=60, step_size=30):
    samples, labels = [], []
    for i in range(0, len(data) - window_size, step_size):
        samples.append(data[i:i+window_size])
        labels.append(targets[i:i+window_size])  
    return np.array(samples), np.array(labels)

