import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import random
import timeit
from numpy import genfromtxt


file_data = []
# reading all the 5 data sets of train and test
phi_x_train_100_10 = genfromtxt("train-100-10.csv",delimiter=',')
t_train_100_10 = genfromtxt("trainR-100-10.csv",delimiter=',')
phi_x_test_100_10 = genfromtxt("test-100-10.csv",delimiter=',')
t_test_100_10 = genfromtxt("testR-100-10.csv",delimiter=',')

phi_x_train_100_100 = genfromtxt("train-100-100.csv",delimiter=',')
t_train_100_100 = genfromtxt("trainR-100-100.csv",delimiter=',')
phi_x_test_100_100 = genfromtxt("test-100-100.csv",delimiter=',')
t_test_100_100 = genfromtxt("testR-100-100.csv",delimiter=',')

phi_x_train_1000_100 = genfromtxt("train-1000-100.csv",delimiter=',')
t_train_1000_100 = genfromtxt("trainR-1000-100.csv",delimiter=',')
phi_x_test_1000_100 = genfromtxt("test-1000-100.csv",delimiter=',')
t_test_1000_100 = genfromtxt("testR-1000-100.csv",delimiter=',')

phi_x_train_crime = genfromtxt("train-crime.csv",delimiter=',')
t_train_crime = genfromtxt("trainR-crime.csv",delimiter=',')
phi_x_test_crime = genfromtxt("test-crime.csv",delimiter=',')
t_test_crime = genfromtxt("testR-crime.csv",delimiter=',')

phi_x_train_wine = genfromtxt("train-wine.csv",delimiter=',')
t_train_wine = genfromtxt("trainR-wine.csv",delimiter=',')
phi_x_test_wine = genfromtxt("test-wine.csv",delimiter=',')
t_test_wine = genfromtxt("testR-wine.csv",delimiter=',')

# List of all the files read
file_data = [[phi_x_train_100_10,t_train_100_10, phi_x_test_100_10, t_test_100_10],
             [phi_x_train_100_100,t_train_100_100, phi_x_test_100_100, t_test_100_100],
             [phi_x_train_1000_100,t_train_1000_100, phi_x_test_1000_100, t_test_1000_100],
             [phi_x_train_crime,t_train_crime, phi_x_test_crime, t_test_crime],
             [phi_x_train_wine,t_train_wine,phi_x_test_wine,t_test_wine]]


#this function calculates the mean square error
def mean_square_error(x, y, w):
    sum = 0
    for i in range(len(y)):
        sum += ((np.dot(x[i].T,w) - y[i])**2)
    return sum/len(y)

#this function finds the w value using equation 3.28
def calculate_w_parameter(x, y, lamda):
    w_list = list()
    for i in range(lamda):
        temp = np.dot(x.T, x)
        a = temp + i * np.identity(len(temp[0]))
        #print(np.shape(a),a)
        w = np.dot(np.dot(np.linalg.inv(a), x.T), y)
        #print("w:",w)
        w_list.append(w)
    return w_list

#Function to iterate over regularization parameter from 0 to 150 and calculate we and mse for the dataset
def call_to_findW_MSE(data):
    lamda = 151
    w = calculate_w_parameter(data[0], data[1], lamda)
    phix = []
    phixr = []
    for i in w:
        trainValues = mean_square_error(data[0], data[1], i)
        testValues = mean_square_error(data[2], data[3], i)
        phix.append(trainValues)
        phixr.append(testValues)
    return phix, phixr

# Function to plot graphs for each dataset with MSE as a function of Lambda for train and test
def plot_graphs_MSE_train_test(train_mse, test_mse, file_train, file_test):
    lam = list(range(151))
    plt.figure()
    plt.plot(lam, train_mse, 'tab:red', label="" + file_train)
    plt.plot(lam, test_mse, 'tab:green', label="" + file_test)
    plt.legend()
    plt.title("MSE as a function of Lambda for test and train")
    plt.ylabel("MSE")
    plt.xlabel("Lambda value")
    plt.savefig('' + file_train + '.png')

print('Task 1 in progress')

#getting MSEs of all files in the list
phix_100_10, phixr_100_10 = call_to_findW_MSE(file_data[0])
phix_100_100, phixr_100_100 = call_to_findW_MSE(file_data[1])
phix_1000_100, phixr_1000_100 = call_to_findW_MSE(file_data[2])
phix_crime, phixr_crime = call_to_findW_MSE(file_data[3])
phix_wine, phixr_wine = call_to_findW_MSE(file_data[4])

# Printing for each dataset minimum value of MSE found as lambda
print("\nDataset -100-10:")
print("Minimum MSE found at lamda " ,np.argmin(phixr_100_10), " = " , phixr_100_10[np.argmin(phixr_100_10)] )

print("\nDataset -100-100:")
print("Minimum MSE found at lamda " ,np.argmin(phixr_100_100), " = " , phixr_100_100[np.argmin(phixr_100_100)] )

print("\nDataset -1000-100:")
print("Minimum MSE found at lamda " ,np.argmin(phixr_1000_100), " = " , phixr_1000_100[np.argmin(phixr_1000_100)] )

print("\nDataset - Crime:")
print("Minimum MSE found at lamda " ,np.argmin(phixr_crime), " = " , phixr_crime[np.argmin(phixr_crime)] )

print("\nDataset - Wine:")
print("Minimum MSE found at lamda " ,np.argmin(phixr_wine), " = " , phixr_wine[np.argmin(phixr_wine)] )

# Calls to plot graphs
plot_graphs_MSE_train_test(phix_100_10, phixr_100_10, 'train-100-10.csv','test-100-10.csv' )
plot_graphs_MSE_train_test(phix_100_100, phixr_100_100, 'train-100-100.csv','test-100-100.csv')
plot_graphs_MSE_train_test(phix_1000_100, phixr_1000_100, 'train-1000-100.csv','test-1000-100.csv')
plot_graphs_MSE_train_test(phix_crime, phixr_crime, 'train_crime.csv', 'test_crime.csv' )
plot_graphs_MSE_train_test(phix_wine, phixr_wine, 'train_wine.csv','test_wine.csv')

print('Task 1 completed and graphs have been saved')
##############################TASK 2###########################

train_size = list(range(10, 801,10))
#Assume
small = 5
# Minimum value of MSE for lmabda 27 from part 1
perfect = 27
#Assume
large = 120

# Fucntion to generate random samples
def mse_random_sample(l):
    all_mse = list()
    for s in train_size:
        all_mse.append(multiple_runs(l, s))
    return all_mse

#l is the lambda value the regularization parameter we need
def find_w_for_given_lamda(x, t, l):
    temp = np.dot(x.T, x)
    a = temp + l * np.identity(len(temp[0]))
    w = np.dot(np.dot(np.linalg.inv(a), x.T), t)
    return w

#random sampling
def random_samples(l):
    train = []
    label = []
    for i in random.sample(range(0, 1000), l):
        train.append(phi_x_train_1000_100[i])
        label.append(t_train_1000_100[i])
    return (np.array(train), np.array(label))

#Performing the task 10 times and taking the average
def multiple_runs(z, k):
    sum = 0
    for i in range(10):
        sampled_train, sampled_label = random_samples(k)
        w = find_w_for_given_lamda(sampled_train, sampled_label, z)
        mse = mean_square_error(phi_x_test_1000_100, t_test_1000_100, w)
        sum += mse
    return sum / 10

print('Task 2 in progress ')
mse_small = mse_random_sample(small)
mse_perfect = mse_random_sample(perfect)
mse_big = mse_random_sample(large)

#plotting graphs
fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(15,15))
fig.suptitle("Task 2 : Learning curves", fontsize=16)
ax[0][0].set_title("For Low Lamda = 5")
ax[0][0].plot(train_size, mse_small, 'tab:red', label="lambda = 5")
ax[0][0].set_ylabel('MSE')
ax[0][0].legend()
ax[0][0].set_xlabel('Training Size')
ax[0][0].grid(True)

ax[0][1].set_title("For Perfect Lamda = 27")
ax[0][1].plot(train_size, mse_perfect, 'tab:orange', label="lambda = 27")
ax[0][1].set_ylabel('MSE')
ax[0][1].legend()
ax[0][1].set_xlabel('Training')
ax[0][1].grid(True)

ax[1][0].set_title("For Large Lamda = 120")
ax[1][0].plot(train_size, mse_big, 'tab:green', label="lambda = 120")
ax[1][0].set_ylabel('MSE')
ax[1][0].legend()
ax[1][0].set_xlabel('Training Size')
ax[1][0].grid(True)

ax[1][1].set_title("For all Lamdas")
ax[1][1].plot(train_size, mse_small, 'tab:red', label="lambda = 5")
ax[1][1].plot(train_size, mse_perfect, 'tab:orange', label="lambda = 27")
ax[1][1].plot(train_size, mse_big, 'tab:green', label="lambda = 120")
ax[1][1].legend()
ax[1][1].set_ylabel('MSE')
ax[1][1].set_xlabel('Training Size')
ax[1][1].grid(True)

fig.savefig('Task 2')

print('Task 2 completed and graphs have been saved')

#########################################PART 3.1#########################################
print('Task 3.1 10 fold cross validation for model selection started:')

# Function to create folds
def fold_i_of_k(dataset, i, k):
    n = len(dataset)
    return dataset[n*(i-1)//k:n*i//k]

strt_time_100_10 = timeit.default_timer()
splits = 11
list_100_10=[]
for lamda in range(0,151):
    sum = 0
    for i in range(1,splits):
        test_index_100_10 = fold_i_of_k(list(range(0, len(phi_x_train_100_10))), i, 10)
        train_index_100_10 = [x for x in range(len(phi_x_train_100_10)) if x not in test_index_100_10]
        X_train_100_10, X_test_100_10 = phi_x_train_100_10[train_index_100_10], phi_x_train_100_10[test_index_100_10]
        T_train_100_10, T_test_100_10 = t_train_100_10[train_index_100_10], t_train_100_10[test_index_100_10]
        w = find_w_for_given_lamda(X_train_100_10, T_train_100_10, lamda)
        mse1 = mean_square_error(X_test_100_10, T_test_100_10, w)
        sum += mse1

    avg1=(sum / 10)
    list_100_10.append(avg1)

lambda_100_10=list_100_10.index(min(list_100_10))
w_100_10=find_w_for_given_lamda(phi_x_train_100_10,t_train_100_10,lambda_100_10)
mse_100_10 = mean_square_error(phi_x_test_100_10,t_test_100_10,w_100_10)
elpsed_100_10 = timeit.default_timer() - strt_time_100_10
print("---------------------------------")
print("Results for 100 10 dataset: ")
print("Lamda : ",lambda_100_10)
print("MSE : ",mse_100_10)
print('Run Time :',elpsed_100_10)


strt_time_100_100 = timeit.default_timer()
list_lamda_100_100=[]
for lamda in range(0,151):
    sum = 0
    for i in range(1, splits):
        test_index_100_100 = fold_i_of_k(list(range(0, len(phi_x_train_100_100))), i, 10)
        train_index_100_100 = [x for x in range(len(phi_x_train_100_100)) if x not in test_index_100_100]
        X_train_100_100, X_test_100_100 = phi_x_train_100_100[train_index_100_100], phi_x_train_100_100[test_index_100_100]
        T_train_100_100, T_test_100_100 = t_train_100_100[train_index_100_100], t_train_100_100[test_index_100_100]
        w = find_w_for_given_lamda(X_train_100_100, T_train_100_100, lamda)
        mse2= mean_square_error(X_test_100_100, T_test_100_100, w)
        sum += mse2
    avg2=(sum / 10)
    list_lamda_100_100.append(avg2)

w_100_100=find_w_for_given_lamda(phi_x_train_100_100,t_train_100_100,np.argmin(list_lamda_100_100))
mse_100_100 = mean_square_error(phi_x_test_100_100,t_test_100_100,w_100_100)
elpsed_100_100 = timeit.default_timer() - strt_time_100_100
print("---------------------------------")
print("Results for 100 100 dataset: ")
print("Lamda : ",np.argmin(list_lamda_100_100))
print("MSE : ",mse_100_100)
print('Run Time :',elpsed_100_100)

strt_time_1000_100 = timeit.default_timer()
list_lamda_1000_100=[]
for lamda in range(0,151):
    sum = 0
    for i in range(1, splits):
        test_index_1000_100 = fold_i_of_k(list(range(0, len(phi_x_train_1000_100))), i, 10)
        train_index_1000_100 = [x for x in range(len(phi_x_train_1000_100)) if x not in test_index_1000_100]
        X_train_1000_100, X_test_1000_100 = phi_x_train_1000_100[train_index_1000_100], phi_x_train_1000_100[test_index_1000_100]
        T_train_1000_100, T_test_1000_100 = t_train_1000_100[train_index_1000_100], t_train_1000_100[test_index_1000_100]

        w = find_w_for_given_lamda(X_train_1000_100, T_train_1000_100, lamda)
        mse3 = mean_square_error(X_test_1000_100, T_test_1000_100, w)
        sum += mse3
    avg3=(sum / 10)
    list_lamda_1000_100.append(avg3)

w_1000_100=find_w_for_given_lamda(phi_x_train_1000_100,t_train_1000_100,np.argmin(list_lamda_1000_100))
mse_1000_100 = mean_square_error(phi_x_test_1000_100,t_test_1000_100,w_1000_100)
elpsed_1000_100 = timeit.default_timer() - strt_time_1000_100
print("---------------------------------")
print("Results for 1000 100 dataset: ")
print("Lamda : ",np.argmin(list_lamda_1000_100))
print("MSE : ",mse_1000_100)
print('Run Time :',elpsed_1000_100)

strt_time_crime = timeit.default_timer()
list_crime=[]
for lamda in range(0,151):
    sum = 0
    for i in range(1, splits):
        test_index_crime = fold_i_of_k(list(range(0, len(phi_x_train_crime))), i, 10)
        train_index_crime = [x for x in range(len(phi_x_train_crime)) if x not in test_index_crime]
        #print(test_index_crime,train_index_crime)
        X_train_crime, X_test_crime = phi_x_train_crime[train_index_crime], phi_x_train_crime[test_index_crime]
        T_train_crime, T_test_crime = t_train_crime[train_index_crime], t_train_crime[test_index_crime]

        w = find_w_for_given_lamda(X_train_crime, T_train_crime, lamda)
        mse4 = mean_square_error(X_test_crime, T_test_crime, w)
        sum += mse4
    avg4=(sum / splits)
    list_crime.append(avg4)

w_crime=find_w_for_given_lamda(phi_x_train_crime,t_train_crime,np.argmin(list_crime))
mse_crime = mean_square_error(phi_x_test_crime,t_test_crime,w_crime)
elpsed_crime = timeit.default_timer() - strt_time_crime
print("---------------------------------")
print("Results for Crime dataset: ")
print("Lamda : ",np.argmin(list_crime))
print("MSE : ",mse_crime)
print('Run Time :',elpsed_crime)

strt_time_wine = timeit.default_timer()
list_wine=[]
for lamda in range(0,151):
    sum = 0
    for i in range(1, splits):
        test_index_wine = fold_i_of_k(list(range(0, len(phi_x_train_wine))), i, 10)
        train_index_wine = [x for x in range(len(phi_x_train_wine)) if x not in test_index_wine]
        X_train_wine, X_test_wine = phi_x_train_wine[train_index_wine], phi_x_train_wine[test_index_wine]
        T_train_wine, T_test_wine = t_train_wine[train_index_wine], t_train_wine[test_index_wine]

        w = find_w_for_given_lamda(X_train_wine, T_train_wine, lamda)
        mse5 = mean_square_error(X_test_wine, T_test_wine, w)
        sum += mse5
    avg5=(sum / 10)
    list_wine.append(avg5)

#print(list_wine)
w_wine=find_w_for_given_lamda(phi_x_train_wine,t_train_wine,np.argmin(list_wine))
mse_wine = mean_square_error(phi_x_test_wine,t_test_wine,w_wine)
elpsed_wine = timeit.default_timer() - strt_time_wine
print("---------------------------------")
print("Results for Wine dataset: ")
print("Lamda : ",np.argmin(list_wine))
print("MSE : ",mse_wine)
print('Run Time :',elpsed_wine)
print('\n')
print('Part 3.1 completed')
print('\n')
#########################################PART 3.2##########################################
#Using iterative approach to iterate the process for 40 runs
def iterative_alpha_beta(phix, t, iterations):
    alpha = random.randrange(1,10)
    beta = random.randrange(1,10)

    phimatrix = (phix.T).dot(phix)
    eigen_values, v = LA.eig(phimatrix)

    N=len(t)

    #Using equation 3.53 and 3.54
    sn_i = alpha * (np.identity(len(phimatrix))) + beta * phimatrix
    sn = np.linalg.inv(sn_i)
    mn = beta * (np.dot(sn, (np.dot(phix.T, t))))


    for i in range(iterations):
        gamma = 0
        #Calculate gamma value
        for eig in eigen_values:
            ##Using equation 3.91
            gamma += eig  / (eig + alpha)

        #Using equation 3.92
        alpha = gamma / (np.dot(mn.T, mn))

        #Using Equation 3.95
        sum = 0
        for i in range(N):
            sum += math.pow((t[i] - np.dot(mn.T, phix[i])), 2)
        beta = (N - gamma) / sum

        #Calculate sn and mn using alpha and beta
        sn_i = alpha * (np.identity(len(phimatrix))) + beta * phimatrix
        sn = np.linalg.inv(sn_i)
        mn = beta * (np.dot(sn, np.dot((phix.T), t)))

    return mn, alpha, beta, alpha/beta, sn

print('Task 3.2 in progress')

# Calls for each dataset to calculate mse and w
start_time_100_10 = timeit.default_timer()
mn_100_10,alpha_100_10,beta_100_10, lamda_100_10, sn_100_10 = iterative_alpha_beta(phi_x_train_100_10,t_train_100_10,40)
w_100_10=find_w_for_given_lamda(phi_x_train_100_10,t_train_100_10, lamda_100_10)
mse_100_10=mean_square_error(phi_x_test_100_10,t_test_100_10,w_100_10)
elapsed_100_10 = timeit.default_timer() - start_time_100_10

start_time_100_100 = timeit.default_timer()
mn_100_100,alpha_100_100,beta_100_100, lamda_100_100,sn_100_100 = iterative_alpha_beta(phi_x_train_100_100,t_train_100_100,40)
w_100_100=find_w_for_given_lamda(phi_x_train_100_100,t_train_100_100, lamda_100_100)
mse_100_100=mean_square_error(phi_x_test_100_100,t_test_100_100,w_100_100)
elapsed_100_100 = timeit.default_timer() - start_time_100_100

start_time_1000_100 = timeit.default_timer()
mn_1000_100,alpha_1000_100,beta_1000_100, lamda_1000_100, sn_1000_100 = iterative_alpha_beta(phi_x_train_1000_100,t_train_1000_100,40)
w_1000_100=find_w_for_given_lamda(phi_x_train_1000_100,t_train_1000_100, lamda_1000_100)
mse_1000_100=mean_square_error(phi_x_test_1000_100,t_test_1000_100,w_1000_100)
elapsed_1000_100 = timeit.default_timer() - start_time_1000_100

crime_start_time = timeit.default_timer()
mn_crime,alpha_crime,beta_crime, lamda_crime, sn_crime = iterative_alpha_beta(phi_x_train_crime,t_train_crime,40)
w_crime=find_w_for_given_lamda(phi_x_train_crime,t_train_crime, lamda_crime)
mse_crime=mean_square_error(phi_x_test_crime,t_test_crime,w_crime)
crime_elapsed = timeit.default_timer() - crime_start_time

wine_start_time = timeit.default_timer()
mn_wine,alpha_wine,beta_wine, lamda_wine, sn_wine = iterative_alpha_beta(phi_x_train_wine,t_train_wine,40)
w_wine=find_w_for_given_lamda(phi_x_train_wine,t_train_wine, lamda_wine)
mse_wine=mean_square_error(phi_x_test_wine,t_test_wine,w_wine)
wine_elapsed = timeit.default_timer() - wine_start_time
print("---------------------------------")
print("Results for 100 10 dataset: ")
print("Alpha : ", alpha_100_10)
print("Beta : ",beta_100_10)
print("Lamda : ",lamda_100_10)
print("MSE : ",mse_100_10)
#print('Standard deviation : ',sn_100_10)
print('Run Time :',elapsed_100_10)
print("---------------------------------")
print("Results for 100 100 dataset: ")
print("Alpha : ", alpha_100_100)
print("Beta : ",beta_100_100)
print("Lamda : ",lamda_100_100)
print("MSE : ",mse_100_100)
#print('Standard deviation : ',sn_100_100)
print('Run Time :',elapsed_100_100)
print("---------------------------------")
print("Results for 1000 100 dataset: ")
print("Alpha : ", alpha_1000_100)
print("Beta : ",beta_1000_100)
print("Lamda : ",lamda_1000_100)
print("MSE : ",mse_1000_100)
#print('Standard deviation : ',sn_1000_100)
print('Run Time :',elapsed_1000_100)
print("---------------------------------")
print("Results for Crime dataset: ")
print("Alpha : ", alpha_crime)
print("Beta : ",beta_crime)
print("Lamda : ",lamda_crime)
print("MSE : ",mse_crime)
#print('Standard deviation : ',sn_crime)
print('Run Time :',crime_elapsed)
print("---------------------------------")
print("Results for Wine dataset: ")
print("Alpha : ", alpha_wine)
print("Beta : ",beta_wine)
print("Lamda : ",lamda_wine)
print("MSE : ",mse_wine)
#print('Standard deviation : ',sn_wine)
print('Run Time :',wine_elapsed)
print("---------------------------------")

print('Task 3.2 completed')