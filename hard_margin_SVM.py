import numpy as np
import cvxopt as opt

'''
Description of QP solver - cvxopt
=================
optimal u <--- QP(Q,p,B,b,A,c)
objective function: min 1/2*u.T*Q*u+p.T*u
subject to B*u <= b, A*u = c
'''

'''
Hard margin linear SVM
======================
x,y: array
x shape: Nxd
y shape: Nx1

For hard margin linear SVM:

u = [[b],[w]] shape:(d+1)x1
Q = [[0,0],[0,I]] shape:dxd
p = 0 shape:(d+1)x1
A = y[1,x] shape:Nx(d+1)
c = 1 shape:Nx1
'''

def hard_margin_linear_SVM(x,y):
	N,d = x.shape[0],x.shape[1]
	diag = np.insert(np.ones(d),0,0)
	Q = np.diag(diag)
	p = np.zeros(d+1)[:,np.newaxis]
	A = y*np.insert(x,0,1,axis = 1)
	c = np.ones(N)[:,np.newaxis]
	ans = opt.solvers.qp(opt.matrix(Q,tc='d'),opt.matrix(p,tc='d'),opt.matrix(-A,tc='d'),opt.matrix(-c,tc='d'))
	return ans['x']

'''
Hard margin kernel SVM
======================
x,y: array
x shape: Nxd
y shape: Nx1 

For hard margin kernel SVM:

alpha shape:Nx1
Q[n,m] = yn*ym*kernel(xn,xm) shape:NxN
p = [[-1],[-1],...,[-1]] shape:Nx1
B = -I shape: NxN
b = [[0],[0],...,[0]] shape: Nx1
A = y.T shape:1xN
c = 0

w = sum(alpha_n*yn*zn) for n support vectors
b = yn-w.T*zn
'''

def hard_margin_kernel_SVM(x,y):
	N = x.shape[0]
	d = x.shape[1]
	Q = np.zeros((N,N))
	for n in range(N):
		for m in range(N):
			Q[n,m] = y[n]*y[m]*kernel(x[n],x[m])
	p = -np.ones(N)[:,np.newaxis]
	A = y.T
	c = 0
	B = -np.identity(N)
	b = np.zeros(N)[:,np.newaxis]

	sol = opt.solvers.qp(opt.matrix(Q,tc='d'),opt.matrix(p,tc='d'),opt.matrix(B,tc='d'),opt.matrix(b,tc='d'),opt.matrix(A,tc='d'),opt.matrix(c,tc='d')) 
	alpha = np.asarray(sol['x'])
	return alpha

def kernel(xn,xm):
	return (1+np.dot(xn,xm))**2

'''
Answer to Question 3: according to complementary slackness, sv with alpha>0
Answer to Question 4: hypothesis g(x) = sign(decision boundary)
'''
if (0):
	x = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
	y = np.array([-1,-1,-1,1,1,1,1])[:,np.newaxis]
	#calculate alpha
	alpha = hard_margin_kernel_SVM(x,y)
	#find support vectors, calculate b 
	sv_index = np.where(np.absolute(alpha)>10**(-5))[0]
	sv = x[sv_index]
	b = y[sv_index[0]]-np.sum(alpha[i]*y[i]*kernel(x[i],x[sv_index[0]]) for i in range(len(alpha)))

	print alpha,sv
	print b 