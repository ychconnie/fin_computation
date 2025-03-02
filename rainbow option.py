# -*- coding: utf-8 -*-
"""FC_HW3 amend .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xeEtOrHQk3x5igwtVt-bmc1cSt8OZsE_
"""

import math as math
import numpy as np
import scipy.stats as stats
from numpy.lib.function_base import average

'''
#test data 1
K = 100
R = 0.1
T = 0.5
n = 2 # number of assets
S0 = [95, 95] # 各資產 S0 list
q = [0.05, 0.05] # 各資產 divident list
sigma = [0.5, 0.5] # 各資產 sigma list
cor = np.array([[1, 1],[1, 1]]) # matrix
'''

'''
#test data 2
K = 100
R = 0.1
T = 0.5
n = 2 # number of assets
S0 = [95, 95] # 各資產 S0 list
q = [0.05, 0.05] # 各資產 divident list
sigma = [0.5, 0.5] # 各資產 sigma list
cor = np.array([[1, -1],[-1, 1]]) # matrix
'''

#test data 3
K = 100
R = 0.1
T = 0.5
n = 5 # number of assets
S0 = [95, 95, 95, 95, 95] # 各資產 S0 list
q = [0.05, 0.05, 0.05, 0.05, 0.05] # 各資產 divident list
sigma = [0.5, 0.5, 0.5, 0.5, 0.5] # 各資產 sigma list
cor = np.array([[1, 0.5, 0.5, 0.5, 0.5],[0.5, 1, 0.5, 0.5, 0.5],[0.5, 0.5, 1, 0.5, 0.5],[0.5, 0.5, 0.5, 1, 0.5],[0.5, 0.5, 0.5, 0.5, 1]]) # matrix

#計算C matrix
C = np.zeros((n,n))
for i in range(n):
  for j in range(n):
    C[i][j] = T * sigma[i] * sigma[j] * cor[i][j]
  # a+=1
C

#C=A(T)*A,計算A matrix
A = np.zeros((n,n))
A[0][0] = (C[0][0])**(1/2)
#print(A[0][0])
#計算a1j
for j in range(1,n):
  A[0][j] = C[0][j] / A[0][0]

for i in range(1,n):
  a=0
  for k in range(i):
    a += (A[k][i])**2
    #print(a)
  A[i][i] = math.sqrt(C[i][i] - a)
  aij=0
  for j in range(i+1,n):
    for k in range(0,i):
      aij+= A[k][i] * A[k][j]
    A[i][j] = (C[i][j] - aij) / A[i][i]
    aij=0
akn=0
for k in range(n-1):
  akn += (A[k][n-1])**2
A[n-1][n-1] = (C[n-1][n-1] - akn)**(1/2)
#  for i in range(n):
#   A[n][n] = C[n][n]
print(A)

#計算mean
mu_list = []
for i in range(n):
  mu = math.log(S0[i])+(R-q[i]-((sigma[i]**2)/2))*T
  #mu = math.e**mu
  mu_list.append(mu)
print(mu_list)

"""##basic"""

sample_20=[]

for count in range(20):
  samplelog = np.random.normal(0, 1, 10000)
  for c in range(n-1):
    samplelog = np.vstack((samplelog,np.random.normal(0, 1, 10000)))

  r_ = np.matmul(samplelog.T,A).T

  lnST = np.zeros((n,10000))
  ST = np.zeros((n,10000))
  payoff = np.zeros((n,10000))
  #payoff = []
  for i in range(n):
    for j in range(10000):
      lnST[i][j] = r_[i][j]+mu_list[i]
      ST[i][j] = math.e**(lnST[i][j])

  maxST = []
  payoff=[]
  #print('ST:')
  #print(ST)
  ST_=ST.transpose()
  #print(ST_)
  for i in range(len(ST_)):
    maxST.append(max(ST_[i]))
  for i in range(len(maxST)):
    if maxST[i] > K:
      maxST[i]=(maxST[i]-K)
    else:
      maxST[i]=0

  sample_20.append(average(maxST)*math.e**(-R*T))

mean = average(sample_20)
sd = np.std(sample_20)
print('Option value:',mean)
print(sd)
print('basic 95% confidence interval:['+str(mean-2*sd)+', '+str(mean+2*sd)+']')

"""##bonus 1 & 2

"""

count=0
#n=4
sample_20=[]
sample20=[]

for count in range(20):
  samplelog = np.random.normal(0, 1, 10000)
  for c in range(n-1):
    samplelog = np.vstack((samplelog,np.random.normal(0, 1, 10000)))

  for i in range(n):
    for j in range(5000,10000):
      samplelog[i][j] = -samplelog[i][j-5000]

  sample1=np.zeros((n,10000))
  for i in range(n):
    sample1[i] = samplelog[i] / np.std(samplelog[i])
  #print(samplelog)

  r_ = np.matmul(A.T,sample1)

  lnST = np.zeros((n,10000))
  ST = np.zeros((n,10000))
  payoff = np.zeros((n,10000))
  #payoff = []
  for i in range(n):
    for j in range(10000):
      lnST[i][j] = r_[i][j]+mu_list[i]
      ST[i][j] = math.e**(lnST[i][j])

  maxST_ = []
  ST_=ST.transpose()
  #print(ST_)
  for i in range(len(ST_)):
    maxST_.append(max(ST_[i]))
  for i in range(len(maxST_)):
    if maxST_[i] > K:
      maxST_[i]=(maxST_[i]-K)
    else:
      maxST_[i]=0
  sample_20.append(average(maxST_)*math.e**(-R*T))

  sample = np.zeros((n,10000))
  for i in range(n):
    for j in range(10000):
      sample[i][j] = samplelog[i][j]

  samplelog_=np.zeros((n,10000))
  for i in range(n):
    for j in range(10000):
      samplelog_[i][j] = sample[i][j] - average(sample[i])

    #print(average(samplelog[i]))
  #print(samplelog_)
  Cnew = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      Cnew[i][j] = np.cov(samplelog_[i],samplelog_[j])[0][1]
  #print(Cnew)

  Anew = np.zeros((n,n))
  Anew[0][0] = (Cnew[0][0])**(1/2)
  #print(A[0][0])
  #計算a1j
  for j in range(1,n):
    Anew[0][j] = Cnew[0][j] / Anew[0][0]

  for i in range(1,n):
    a=0
    for k in range(i):
      a += (Anew[k][i])**2
      #print(a)
    Anew[i][i] = math.sqrt(Cnew[i][i] - a)
    aij=0
    for j in range(i+1,n):
      for k in range(0,i):
        aij+= Anew[k][i] * Anew[k][j]
      Anew[i][j] = (Cnew[i][j] - aij) / Anew[i][i]
      aij=0
  akn=0
  for k in range(n-1):
    akn += (Anew[k][n-1])**2
  Anew[n-1][n-1] = (Cnew[n-1][n-1] - akn)**(1/2)

  #print(Anew)
  #print(np.matmul(Anew.T,Anew))
  invAnew = np.linalg.inv(Anew)

 #print(invAnew)
  #print(np.matmul(Anew,invAnew))
  Z = np.matmul(invAnew.T,samplelog_)
  # print('Z:')
  # print(Z)
  r = np.matmul(Z.T,A).T
  #print('r:')
  #print(r)

  lnST2 = np.zeros((n,10000))
  ST2 = np.zeros((n,10000))
  for i in range(n):
    for j in range(10000):
      lnST2[i][j] = r[i][j]+mu_list[i]
      ST2[i][j] = math.e**(lnST2[i][j])

  maxST2 = []
  #payoff2=[]

  ST_2=ST2.transpose()
  #print(ST_)
  for i in range(len(ST_2)):
    maxST2.append(max(ST_2[i]))
  for i in range(len(maxST2)):
    if maxST2[i] > K:
      maxST2[i]=(maxST2[i]-K)
    else:
      maxST2[i]=0

  sample20.append(average(maxST2)*math.e**(-R*T))

#print(sample_20)
#print(average(sample_20))
#print(np.std(sample_20))
mean = average(sample_20)
#print('Option Value:',mean)
sd = np.std(sample_20)
print(sd)
print('bonus1 95% confidence interval:['+str(mean-2*sd)+', '+str(mean+2*sd)+']')


mean2 = average(sample20)
#print(mean2)
sd2 = np.std(sample20)
print(sd2)
print('bonus2 95% confidence interval:['+str(mean2-2*sd2)+', '+str(mean2+2*sd2)+']')
#print(sd-sd2)