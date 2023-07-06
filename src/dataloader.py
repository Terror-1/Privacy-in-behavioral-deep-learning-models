import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, plot_model


db=pd.read_pickle("../datasets/sample.pkl")
mouse_data = db[db['type'].isin(['mousemove', 'mousedown', 'mouseup'])]
keyboard_data = db[db['type'].isin(['keyup', 'keydown'])]
keyboard_data=keyboard_data.drop(['ID', 'INVALID', 'X', 'Y', 'resolutionX', 'resolutionY', 'mu'], axis=1)
mouse_data=mouse_data.drop(['ID', 'INVALID', 'type', 'value', 'mu'], axis=1)
keystrokes = keyboard_data['value'].unique()
def load_mouse_data():
    USERS = set(mouse_data['user'])
    X_train = pd.DataFrame(columns = ['O','C','E','A','N'])
    X_test  = pd.DataFrame(columns = ['O','C','E','A','N'])
    y_train = pd.DataFrame()
    y_test  = pd.DataFrame()
    for index,user in enumerate(USERS) :
     X_user= mouse_data[mouse_data['user']==user]
     X = X_user[['O','C','E','A','N']]
     y = X_user['user']
     X_train_user, X_test_user, y_train_user, y_test_user = train_test_split(X, y, test_size=0.2, random_state=42)
     X_train = pd.concat([X_train,X_train_user])
     X_test= pd.concat([X_test,X_test_user])
     y_train = pd.concat([y_train,y_train_user])
     y_test= pd.concat([y_test,y_test_user])
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return  X_train.astype('float32'), X_test.astype('float32'),y_train.astype('float32'), y_test.astype('float32')