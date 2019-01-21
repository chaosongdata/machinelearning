#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:30:28 2019

@author: c0s02bi
"""

'''
we would like to estimate the bayesian linear regression with variance inference.


'''
import numpy as np
import matplotlib.pyplot as plt

class bayessianregression():
    def __init__(self,train,train_label,bound,name='bayessianregression'):
        self.name = name
        self.train = train
        self.label = train_label
        self.n = self.train.shape[0]
        self.d = self.train.shape[1]
        self.bound = bound
        self.N = 100
        self.dom = np.linspace(-bound,bound,self.N)
        self.coeff = np.random.normal(0,1,[self.d,len(self.dom)])  
        self.coeff[0,:] = (1/np.power(2*np.pi,0.5))*np.exp(-0.5*np.power(self.dom,2)) 
        self.coeff[1,:] = (1/np.power(2*np.pi,0.5))*np.exp(-0.5*np.power(self.dom,2)) # store the distribution of coeff
    
    def renormalize(self,t):
        s = sum(t)
        t = t/s
        return t
    
    def DLdiv(self,p,q):
        return abs(sum(np.log(p/q)*p))
    
    def update(self,i,j):
        print(sum(0.5*np.power((self.train[j,1]*self.dom-self.label[j,0]+self.coeff[0,i]*self.train[j,0]),2)*self.coeff[1,:]))
    
    def fit(self):
        '''
        use variance inference to update
        '''
        tmp = np.zeros([self.d,len(self.dom)])
        diff = 1000
        iteration = 0
        while (diff > 0.02 and iteration <= 50):
            tmp = np.zeros([self.d,len(self.dom)])
            for i in range(self.N):
                for j in range(0,self.n):
                    tmp[0,i] = tmp[0,i] + (1/self.n)*sum(0.5*np.power((self.train[j,1]*self.dom-self.label[j,0]+self.dom[i]*self.train[j,0]),2)*self.coeff[1,:])
        #self.coeff[0,:] = tmp[0,:]
            tmp[0,:] = np.exp(-tmp[0,:])    
            tmp[0,:] = self.renormalize(tmp[0,:])
            diff = self.DLdiv(tmp[0,:],self.coeff[0,:])
            for i in range(self.N):
                for j in range(self.n):
                    tmp[1,i] = tmp[1,i] + (1/self.n)*sum(0.5*np.power((self.train[j,0]*self.dom-self.label[j,0]+self.dom[i]*self.train[j,1]),2)*tmp[0,:])  
            tmp[1,:] = np.exp(-tmp[1,:])
            tmp[1,:] = self.renormalize(tmp[1,:])
            diff = diff + self.DLdiv(tmp[1,:],self.coeff[1,:])
            iteration = iteration + 1
            self.coeff = tmp
            print(diff)
        return self
    
  

    def predict(self,x):
        '''
        more automatic to deal with d >= 2
        
        '''
        res = 0
        for i in range(len(self.dom)):
            for j in range(len(self.dom)):
                res = res + self.coeff[0,i]*self.coeff[1,j]*(self.dom[i]*x[0]+self.dom[j]*x[1])
        return res

def mape(a,b):
    return np.mean(abs(a-b)/b)*100
    
if __name__ == "__main__":
    x = np.random.normal(0,2,[1000,2])
    theta = np.random.normal(0,1,[2,1])
    print("theta={x}".format(x=theta))
    y = np.dot(x,theta)
    y = y + np.random.normal(0,0.1,[1000,1])
    split = 800
    train_x = x[:split,:] 
    test_x = x[split:,:]
    train_y = y[:split,:]
    test_y = y[split:,:]
    bayes_model = bayessianregression(train_x,train_y,3)
    bayes_model.fit()
    t = bayes_model.predict(test_x[0])
    #print(bayes_model.coeff)
    plt.plot(bayes_model.dom,bayes_model.coeff[0,:],'r-',bayes_model.dom,bayes_model.coeff[1,:],'-')
    p_ls = []
    for i in range(test_x.shape[0]):
        p_ls.append(bayes_model.predict(test_x[i,:]))   
    print(mape(np.asarray(p_ls),test_y[:,0]))    
    
    