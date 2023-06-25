import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.lines import Line2D
import dp_analysis as dp

### data import and split into mouse and keyboard
## pre-processing of keyboard data and mouse data
db=pd.read_pickle("../datasets/sample.pkl")
mouse_data = db[db['type'].isin(['mousemove', 'mousedown', 'mouseup'])]
keyboard_data = db[db['type'].isin(['keyup', 'keydown'])]
keyboard_data=keyboard_data.drop(['ID', 'INVALID', 'X', 'Y', 'resolutionX', 'resolutionY', 'mu'], axis=1)
mouse_data=mouse_data.drop(['ID', 'INVALID', 'type', 'value', 'mu'], axis=1)
keystrokes = keyboard_data['value'].unique()
# encode the keystrokes
# 0 > keydown 1 > keyup
# 0 > 112 keystorkes
for i in range(len(keystrokes)):
    keyboard_data['value'].replace(keystrokes[i], i, inplace=True)

# change the type columns to 0 and 1
keyboard_data['type'].replace('keydown', 0, inplace=True)
keyboard_data['type'].replace('keyup', 1, inplace=True)
# calculate the frequency of each keystroke
keystroke_frequency = keyboard_data['value'].value_counts()


#------------------------------------------------------------------------------------------------------

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

#pds = get_parallel_dbs(db)
## extract senstive data from the dataframe (it is not actually senstive data, just for testing)
# let's say the senstive data is the sum of the resolutionX 
def sum_query(db):
    return mouse_data['X'].sum()

def mean_query(db):
    return db['X'].mean()

full_db_result_sum = sum_query(db)
full_db_result_mean = mean_query(db)
sum_senstivity = 0
mean_senstivity = 0
"""for pdb in pds:
    pdb_result = sum_query(pdb)
    db_distance = abs(pdb_result - full_db_result_sum)
    if db_distance > sum_senstivity:
        sum_senstivity = db_distance

for pdb in pds:
    pdb_result = mean_query(pdb)
    db_distance = abs(pdb_result - full_db_result_mean)
    if db_distance > mean_senstivity:
        mean_senstivity = db_distance
"""        

#senstivity of sum function
print("sum_senstivity: ", sum_senstivity)
print(laplacian_noise_mechanism(sum_senstivity, epislon,sum_query,db))
print("mean_senstivity: ", mean_senstivity)
print(laplacian_noise_mechanism(mean_senstivity, epislon,mean_query,db))
print("--------------------")

        
avg_dp , avg_error , used_eps ,noise= dp.dp_analysis_sum_test(db['X'], 0.01, 100)
avg_dp_mean , avg_error_mean , used_eps_mean ,noise= dp.dp_analysis_mean_test(db['X'], 0.01, 100)

def laplace_noise(scale, epsilon):
    return np.random.laplace(loc=0, scale=scale/epsilon)

sensitivity = 1
epsilons = np.linspace(0.1, 1, 10)
noise_levels = [sensitivity / epsilon for epsilon in epsilons]
"""
plt.plot(epsilons, noise_levels, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Noise Level')
plt.title('Amount of Noise Added for Different Epsilon Values')
plt.grid(True)

plt.show()
"""
"""
gs1 = gs.GridSpec(nrows=2, ncols=2)
figure = plt.gcf()
figure2 = plt.gcf()
ax1 = plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[0,1])
ax3 = plt.subplot(gs1[1,0])
ax4 = plt.subplot(gs1[1,1])

ax1.plot(used_eps,noise, marker='o')
ax1.set_ylabel('Laplace Noise', size=15)
ax1.set_xlabel('Epsilon', size=15)
"""
"""
ax1.plot(used_eps,avg_error, color='red')
ax1.set_ylabel('Average percentage error', size=19)
ax1.set_xlabel('Epsilon', size=19)

ax2.plot(used_eps,avg_dp, color='blue')
ax2.set_ylabel('Average diffprivate results', size=19)
ax2.set_xlabel('Epsilon', size=19)
ax2.hlines(y=sum_query(db), xmin=0, xmax=1, colors='red', linestyles='-', lw=2, label='Real result')
ax2.legend()

ax3.plot(used_eps_mean,avg_error_mean, color='red')
ax3.set_ylabel('Average percentage error ', size=19)
ax3.set_xlabel('Epsilon', size=19)


ax4.plot(used_eps_mean,avg_dp_mean, color='blue')
ax4.set_ylabel('Average diffprivate results', size=19)
ax4.set_xlabel('Epsilon', size=19)
ax4.hlines(y=mean_query(db), xmin=0, xmax=1, colors='red', linestyles='-', lw=2, label='Real result')
ax4.legend()
"""


plt.show()





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
