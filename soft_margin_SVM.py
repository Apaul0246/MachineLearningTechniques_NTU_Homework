import sys
sys.path.append('C:\Users\user\OneDrive\Code\libsvm-3.22\python')
from svmutil import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
LIBSVM implements 'one-against-one' approach for multiclass classification.
Data pre-processing needed for 'one-against-all' approach
'''

def svm_ova_15(df,C):
	x = df.loc[:,1:].to_dict('split')['data']
	y = np.where(df.loc[:,0]==0,1,-1)

	prob = svm_problem(y,x)
	param = svm_parameter('-t 0 -s 0 -h 0 '+'-c '+str(C)) 
	m = svm_train(prob,param)
	support_vector_coefficients = m.get_sv_coef()
	support_vectors = m.get_SV()
	sv_indices = m.get_sv_indices()

	w = np.zeros((np.shape(x)[1]))
	for i in range(len(support_vectors)):
		w += support_vector_coefficients[i]*np.asarray([support_vectors[i][1],support_vectors[i][2]])
	return w

#Homework 1 Question 15
if (0):
	df = pd.read_csv('C:\Users\user\Desktop\\train.csv',header=None)
	w_len = []
	C = [0.000001,0.0001,0.01,1]
	for c in C:
		w = svm_ova_15(df,c)
		length = (w[0]**2+w[1]**2)**0.5
		w_len.append(length)
	plt.plot(np.log10(C),w_len,'ro')
	plt.show()

def svm_ova_16(df,C):
	x = df.loc[:,1:].to_dict('split')['data']
	y = np.where(df.loc[:,0]==8,1,-1)
	prob = svm_problem(y,x)
	param = svm_parameter('-t 1 -g 1 -d 2 -r 1 -h 0 '+'-c '+str(C)) 
	m = svm_train(prob,param)
	p_labels, p_acc, p_vals = svm_predict(y, x, m)
	Ein = p_acc[0]
	return Ein 

#Homework 1 Question 16
if(0):
	df = pd.read_csv('C:\Users\user\Desktop\\train.csv',header=None)
	C = [0.000001,0.0001,0.01,1]
	Ein = []
	for c in C:
		err = svm_ova_16(df,c)
		Ein.append(err)
	plt.plot(np.log10(C),Ein,'ro')
	plt.show()

def svm_ova_17(df,C):
	x = df.loc[:,1:].to_dict('split')['data']
	y = np.where(df.loc[:,0]==8,1,-1)
	prob = svm_problem(y,x)
	param = svm_parameter('-t 1 -g 1 -d 2 -r 1 -h 0 '+'-c '+str(C)) 
	m = svm_train(prob,param)
	support_vector_coefficients = m.get_sv_coef()
	sum = 0.0
	for i in range(len(support_vector_coefficients)):
		sum += abs(support_vector_coefficients[i][0])
	return sum 

#Homework 1 Question 17
if(0):
	df = pd.read_csv('C:\Users\user\Desktop\\train.csv',header=None)
	C = [0.000001,0.0001,0.01,1]
	sum_alpha = []
	for c in C:
		sum = svm_ova_17(df,c)
		sum_alpha.append(sum)
	plt.plot(np.log10(C),sum_alpha,'ro')
	plt.show()	

def svm_ova_18(df,C):
	x = df.loc[:,1:].to_dict('split')['data']
	y = np.where(df.loc[:,0]==0,1,-1)

	prob = svm_problem(y,x)
	param = svm_parameter('-t 2 -s 0 -g 100 -h 0 '+'-c '+str(C)) 
	m = svm_train(prob,param)
	support_vector_coefficients = m.get_sv_coef()
	support_vectors = m.get_SV()
	w = np.zeros((np.shape(x)[1]))
	for i in range(len(support_vectors)):
		w += support_vector_coefficients[i][0]*np.asarray([support_vectors[i][1],support_vectors[i][2]])
	return w

#Homework 1 Question 18
if (0):
	df = pd.read_csv('C:\Users\user\Desktop\\train.csv',header=None)
	C = [0.001,0.01,0.1,1,10]
	distance = []
	for c in C:
		w = svm_ova_18(df,c)
		dis = (w[0]**2+w[1]**2)**(0.5)
		distance.append(dis)
	plt.plot(np.log10(C),distance,'ro')
	plt.show()

def svm_ova_19(df,df_test,g):
	x = df.loc[:,1:].to_dict('split')['data']
	y = np.where(df.loc[:,0]==0,1,-1)

	x_t = df_test.loc[:,1:].to_dict('split')['data']
	y_t = np.where(df_test.loc[:,0]==0,1,-1)

	prob = svm_problem(y,x)
	param = svm_parameter('-t 2 -s 0 -c 0.1 -h 0 '+'-g '+str(g)) 
	m = svm_train(prob,param)

	p_labels, p_acc, p_vals = svm_predict(y_t, x_t, m)
	err = p_acc[0]
	return err

#Homework 1 Question 19
if (0):
	df = pd.read_csv('C:\Users\user\Desktop\\train.csv',header=None)
	df_test = pd.read_csv('C:\Users\user\Desktop\\test.csv',header=None)
	Eout = []
	G = [1,10,100,1000,10000]
	for g in G:
		err = svm_ova_19(df,df_test,g)
		Eout.append(err)
	plt.plot(np.log10(G),Eout,'ro')
	plt.show()