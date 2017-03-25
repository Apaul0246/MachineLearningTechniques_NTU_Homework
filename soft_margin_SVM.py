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

def svm_ova(df,C):
	x = df.loc[:,1:].to_dict('split')['data']
	y = np.where(df.loc[:,0]==0,1,0)

	prob = svm_problem(y,x)
	param = svm_parameter('-t 0 -s 0 -h 0 '+'-c '+str(C)) 
	m = svm_train(prob,param)
	support_vector_coefficients = m.get_sv_coef()
	support_vectors = m.get_SV()
	sv_indices = m.get_sv_indices()

	w = np.zeros((np.shape(x)[1]))
	for i in range(len(support_vectors)):
		w += support_vector_coefficients[i]*np.asarray([support_vectors[0][1],support_vectors[0][2]])
	
	return w

df = pd.read_csv('C:\Users\user\Desktop\\train.csv',header=None)
w_len = []
C = [0.000001,0.0001,0.01,1,100]
for c in C:
	w = svm_ova(df,c)
	length = (w[0]**2+w[1]**2)**0.5
	w_len.append(length)

plt.plot(np.log10(C),w_len,'ro')
plt.show()