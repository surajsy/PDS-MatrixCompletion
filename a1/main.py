import pandas as pd
import numpy as np
from random import seed
from random import randrange
from statistics import mean

# https://machinglearningmastery.com/implement-resampling-methods-scratch-python/
def cross_validation_split(data, folds = 5):
	data_split = list()
	data_copy = list(data)
	fold_size = int(len(data)/folds)

	for i in range(folds):
		fold = list()
		while(len(fold) < fold_size):
			index = randrange(len(data_copy))
			fold.append(data_copy.pop(index))
		data_split.append(fold)

	return data_split,folds

def kfold_train_validate(data,iteration,folds = 5):
	data_split,folds = cross_validation_split(data,folds)
	data_split_copy = list(data_split)
	train_data_split = list()

	for i in range(folds):
		if(iteration == i):
			validation_data = data_split_copy[i]
		else:
			train_data_split.append(data_split_copy[i][:])

	train_data = [e for dim1 in train_data_split for e in dim1]

	return train_data,validation_data

def form_matrix(train_data,dim):
	X = np.zeros(dim)
	user,item,rating = [list(x) for x in zip(*train_data)]
	user = list(map(int,user))
	item = list(map(int,item))
	for u,i,r in zip(user,item,rating):
		X[u][i] = r

	return X

max_count = 1000
epsilon = 0.001
folds = 5
G = list(np.logspace(1,2,num=10))
G.reverse()

dataframe = pd.read_csv(
		"ratings.train",
		sep='\t',
		header = None,
		skiprows = 1,
	)

data = list(map(list,dataframe.to_numpy()))
user,item,rating = list(map(list,dataframe.T.to_numpy()))

m = int(max(user) + 1)
n = int(max(item) + 1)

Z_new = np.random.rand(m,n)
best_model = dict()
current_min = float('inf')

for gamma in G:
	print('Training model for gamma =',gamma)
	E = list()
	for k in range(folds):
		print('k fold cross_validation iteration ' + str(k) + ' started...')
		train_data,validation_data = kfold_train_validate(data,k,folds)
		X = form_matrix(train_data,(m,n))

		Z = Z_new
		PZ = np.zeros((m,n))
		indices = np.where(X == 0)

		count = 0
		while(1):
			PZ[indices] = Z[indices]
			U,S,Vt = np.linalg.svd(np.add(PZ,X),full_matrices = False)
			S[np.where(S - gamma <= 0)] = 0
			Z_new = np.dot(np.dot(U,np.diag(S)),Vt)
			count = count + 1
			if(np.linalg.norm(np.subtract(Z_new,Z),'fro') < epsilon * np.linalg.norm(Z,'fro') or count > max_count):
				break
			Z = Z_new

		user,item,rating = [list(x) for x in zip(*validation_data)]
		error = 0
		for (u,i,r) in zip(user,item,rating):
			error = error + (r - Z[int(u)][int(i)])**2
		E.append(error/len(validation_data))
		print('k fold cross_validation iteration ' + str(k) + ' done')

	mean_error = mean(E)
	print('mean error for gamma = ',gamma,': ',mean_error)
	if(mean_error < current_min):
		best_model['Z'] = Z
		best_model['gamma'] = gamma
		best_model['error'] = error

np.save('model', best_model['Z'],allow_pickle=False)