import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

db=pd.read_pickle("../datasets/sample.pkl")
db = db[0:5000]
### convert the dataframe to tensor

epislon = 0.1 # privacy budget
# calculate the senstivity of the function
# senstivity = max |f(x) - f(x')| where x and x' are the two data points that differ in one row


# calculate the laplacian noise  f(x) = x + laplacian_noise(senstivity, epislon)
def laplacian_noise_mechanism(senstivity, epislon,query,db):
    scale = senstivity/epislon
    noise = np.random.laplace(0, scale, 1)
    return query(db) + noise

# plot the laplacian noise unnessecary
def plot_laplacian_noise(senstivity, epislon,query,db):
   count, bins, ignored = plt.hist(laplacian_noise_mechanism(db,query,senstivity,epislon), 30, density=True)
   x = np.arange(-8., 8., .01)
   pdf = np.exp(-abs(x-0)/senstivity/epislon)/(2.*senstivity/epislon)
   plt.plot(x, pdf)

# create random parallel dataset 
def get_parallel_db(df,remove_index):
    return df.drop(remove_index)

# create random parallel datasets
def get_parallel_dbs(db):

    parallel_dbs = list()
    
    for i in range(len(db)):
        pdb = get_parallel_db(db, i)
        parallel_dbs.append(pdb)
    
    return parallel_dbs

pds = get_parallel_dbs(db)
## extract senstive data from the dataframe (it is not actually senstive data, just for testing)
# let's say the senstive data is the sum of the resolutionX 
def sum_query(db):
    return db['X'].sum()

def mean_query(db):
    return db['X'].mean()

full_db_result_sum = sum_query(db)
full_db_result_mean = mean_query(db)
sum_senstivity = 0
mean_senstivity = 0
for pdb in pds:
    pdb_result = sum_query(pdb)
    db_distance = abs(pdb_result - full_db_result_sum)
    if db_distance > sum_senstivity:
        sum_senstivity = db_distance

for pdb in pds:
    pdb_result = mean_query(pdb)
    db_distance = abs(pdb_result - full_db_result_mean)
    if db_distance > mean_senstivity:
        mean_senstivity = db_distance

#senstivity of sum function
print("sum_senstivity: ", sum_senstivity)
print(laplacian_noise_mechanism(sum_senstivity, epislon,sum_query,db))
print("mean_senstivity: ", mean_senstivity)
print(laplacian_noise_mechanism(mean_senstivity, epislon,mean_query,db))

"""
noise = laplacian_noise(senstivity, epislon)
print("how much noise added :", noise)
df_senstive = df[(df['resolutionX'] == 1349) & (df['resolutionY'] == 768)]
# we get the count of the senstive data
df_senstive_count = len(df_senstive)
print("Senstive data count: ", df_senstive_count)
# we add laplacian noise to the senstive data

df_senstive_count = df_senstive_count + noise
print("Senstive data count with lablacian noise: ", df_senstive_count)
"""
"""
df_prime_senstive = df_prime[(df_prime['resolutionX'] == 1349) & (df_prime['resolutionY'] == 768)]
df_prime_senstive_count = len(df_prime_senstive)
#print("Senstive data prime count: ", df_senstive_count)
noise = laplacian_noise(senstivity, epislon)
df_prime_senstive_count = df_prime_senstive_count + noise
print("-------------------------------------------")
print("Senstive data prime count  with lablacian noise: ", df_prime_senstive_count)
"""
"""
#calculate privacy pudget
privacy_pudget = np.log(df_senstive_count/df_prime_senstive_count)
print("privacy pudget: ", "%.2f" %privacy_pudget)
"""
"""
# calculate the guassian noise  f(x) = x + guassian_noise(senstivity, epislon)
def guassian_noise(senstivity, epislon):
    scale = senstivity/epislon
    noise = np.random.normal(0, scale, 1)
    return noise


noise = guassian_noise(senstivity, epislon)
print("how much noise added :", noise)
"""
