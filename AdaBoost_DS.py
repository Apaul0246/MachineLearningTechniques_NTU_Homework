from __future__ import division
import numpy as np
import pandas as pd

df = pd.read_csv("C:\Users\user\OneDrive\Code\hw2_adaboost_train.dat",sep='\s+',header=None)
df_test = pd.read_csv("C:\Users\user\OneDrive\Code\hw2_adaboost_test.dat",sep='\s+',header=None)
df['u'] = (np.ones((len(df)))*1.0/len(df))[:,np.newaxis]

def decision_stump(df):
	N = len(df)
	Ein = 1.0

	for x in [0,1]:
		df = df.sort(x)
		idx = list(df.index.values)

		S = [-1,+1]
		theta = []
		for i in range(N-1):
			t = (df[x][idx[i]]+df[x][idx[i+1]])/2
			theta.append(t)
		for s in S:
			for t in theta:
				new_y = []
				for d in range(N):
					y = s*np.sign(df[x][idx[d]]-t)
					new_y.append(y)
				df['J'] = (df[2] != new_y)
				sum_u_incorrect = df[df['J'] == True]['u'].sum()
				eps = sum_u_incorrect/df['u'].sum()	
				if eps <= Ein:
					Ein = eps
					s_best = s
					t_best = t
					i_best = x
					y_best = new_y
					y_label = df[2]
	return y_label,y_best,Ein,s_best,t_best,i_best


def adaboost(df,df_test,iters):
	G = []
	Err = []
	for t in range(iters):
		label_y,y_best,eps,s,t,feature= decision_stump(df)
		r =((1-eps)/eps)**0.5
		alpha_t = np.log(r)
		G.append((s,t,alpha_t,feature))
		Err.append(eps)
		df['u']= [df['u'][j]/r if (y_best == label_y)[j] else df['u'][j]*r for j in range(len(label_y))]
	pres = []
	for i in range(len(df_test)):
		pre = np.sign(np.sum([alpha_t*s*np.sign(df_test[feature][i]-t) for (s,t,alpha_t,feature) in G]))
		pres.append(pre)
	test_err = np.sum(pres != df_test[2])/len(df_test)
	return G,Err,pres,test_err