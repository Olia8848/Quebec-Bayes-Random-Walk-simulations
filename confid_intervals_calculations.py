# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:29:07 2019

@author: Olga Rumyantseva
"""
import pandas   # data analysis
import numpy    # n-dim arrays
import math
import matplotlib
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
import scipy
from scipy import stats

path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'
df = pandas.read_csv(path + 'biodata.csv')
# df = pandas.read_csv('biodata.csv')
data = df.values  # work with values only
NOBSERV = numpy.size(data, 0) # number of rows in our dataframe (number of observations)


patch = data[:,0]   # plot ID (1st column)
year = numpy.array([int(data[k,1]) for k in range(NOBSERV)]) #years converted in integer from float
biomass = data[:,2]
shadow = data[:,3]
value = biomass # this is what we consider now: biomass or shadow
#Change to shadow if you need

logvalue = numpy.array([]) # here will be logarithms of value for steps
interval = numpy.array([]) # increments of years

for observ in range(NOBSERV-1):
    if patch[observ] == patch[observ+1]:
        logvalue = numpy.append(logvalue, math.log(value[observ]))
        interval = numpy.append(interval, year[observ+1] - year[observ])

#Number of pair-observations
NCHANGES = numpy.size(logvalue)  

# v is a vector of all sqrt(u)
v = numpy.array([])
timegaps = numpy.array([])

for j in range(NCHANGES): 
    u = interval[j]
    timegaps = numpy.append(timegaps, u)
    v = numpy.append(v, numpy.sqrt(u))
    
numpy.mean(timegaps) # 10.1
numpy.sqrt(numpy.sum(timegaps)) # 459.384
numpy.sqrt(numpy.dot(v,v))  # 459.384  
 
# Biomass raw               r^{hat} = 0.0387,  sigma (or s) = 0.223
# Biomass centered          r^{hat} = 0.0569,  sigma (or s) = 0.224
# Biomass centered censored r^{hat} = 0.00570, sigma (or s) = 0.224

# Shadow raw               r^{hat} = 0.0337,   sigma (or s) = 0.207
# Shadow centered          r^{hat} = 0.00340,  sigma (or s) = 0.197
# Shadow centered censored r^{hat} = 0.00341,  sigma (or s) = 0.197

########################################################################
######## Confidence intervals with T-statistics t_{n-1, 1-alpha/2}: ####
########################################################################
U = numpy.dot(v,v) # 211034.0
T = 1.645 # T-statistics t_{n-1, 1-alpha/2} = 1.645

rHat = 0.00341
s = 0.197 
 
X1 = rHat - (s/numpy.sqrt(U))*T 
X2 = rHat + (s/numpy.sqrt(U))*T 
print(X1, X2)

# Biomass raw               
# (X1, X2) = (0.03790146390814564, 0.03949853609185436)

# Biomass centered   
# (X1, X2) = (0.05609788302880997, 0.05770211697119003)        
                       

# Biomass centered censored 
# (X1, X2) = (0.004897883028809974, 0.0065021169711900265) 


# Shadow raw               
# (X1, X2) = (0.03295875797751636, 0.034441242022483644)

# Shadow centered         
# (X1, X2) = (0.002694566770873057, 0.004105433229126943)

# Shadow centered censored 
# (X1, X2) = (0.002704566770873057, 0.004115433229126943)

# Biomass raw               r^{hat} = 0.0387,  sigma (or s) = 0.223
# Biomass centered          r^{hat} = 0.0569,  sigma (or s) = 0.224
# Biomass centered censored r^{hat} = 0.00570, sigma (or s) = 0.224

# Shadow raw               r^{hat} = 0.0337,   sigma (or s) = 0.207
# Shadow centered          r^{hat} = 0.00340,  sigma (or s) = 0.197
# Shadow centered censored r^{hat} = 0.00341,  sigma (or s) = 0.197

########################################################################
######## Confidence intervals with  z_{1-alpha/2}: ####################
########################################################################

# for big n   z_{1-alpha/2} approx. equals t_{n-1, 1-alpha/2}

rHat = 0.00341
s = 0.197 

Y1 = ((1 + T/(numpy.sqrt(NOBSERV)))**(-1))*s     
Y2 = ((1 - T/(numpy.sqrt(NOBSERV)))**(-1))*s 
print(Y1, Y2)

# Biomass raw               
# (Y1, Y2) = (0.22098516179389846, 0.22505191695659388)

# Biomass centered   
# (Y1, Y2) = (0.22197612664499217, 0.2260611183779239)        
                       
# Biomass centered censored 
# (Y1, Y2) = (0.22197612664499217, 0.2260611183779239) 


# Shadow raw               
# (Y1, Y2) = (0.205129724176399, 0.20890469421531357)

# Shadow centered         
# (Y1, Y2) = (0.19522007566546187, 0.19881268000201344)

# Shadow centered censored 
# (Y1, Y2) = (0.19522007566546187, 0.19881268000201344)







  
    
