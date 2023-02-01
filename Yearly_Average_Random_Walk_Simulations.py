# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:40:26 2019

@author: Olga Rumyantseva
"""


import pandas   # data analysis
import numpy    # n-dim arrays
import math
import matplotlib
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
import scipy    # lin regress
import matplotlib.pyplot as plt
#from pandas.tools.plotting import table
import random

##################################################################
#################### Preparation Block: ##########################
##################################################################


path = 'C:/Users/olga/Desktop/Quebec/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'

# df = pandas.read_csv(path + 'biodata.csv')
df = pandas.read_csv(path + 'biodata.csv') # header added as the 
# df.columns

data = df.values  # work with values only
NOBSERV = numpy.size(data, 0) # number of rows in our dataframe (number of observations)
print(NOBSERV) 


patch = data[:,0]   # plot ID (1st column)
year = numpy.array([int(data[k,1]) for k in range(NOBSERV)]) #years converted in integer from float
biomass = data[:,2]
shadow = data[:,3]
value = biomass # this is what we consider now: biomass or shadow


##################################################################
#####  Annual means of biomass and shadow: ########################
##################################################################


Years = numpy.unique(year)
NYears = numpy.size(Years)

# Annual means of biomass:
AnnualMeanBiomass = numpy.array([]) 

for i in range(NYears):
    logbio = math.log(numpy.mean(value[year == Years[i]]))
    AnnualMeanBiomass = numpy.append(AnnualMeanBiomass, logbio)
    

fig1 = plt.figure()
pyplot.plot(Years, AnnualMeanBiomass, 'bo', Years, AnnualMeanBiomass, 'k')
pyplot.show()



# Annual means of shadows:
AnnualMeanShadow = numpy.array([]) 

for i in range(NYears):
    logshad = math.log(numpy.mean(shadow[year == Years[i]]))
    AnnualMeanShadow = numpy.append(AnnualMeanShadow, logshad)
    

##################################################################
#####  Plots of annual means of biomass and shadow: ###############
##################################################################



pyplot.plot(Years, AnnualMeanBiomass, 'bo', Years, AnnualMeanBiomass, 'k')
pyplot.show()

pyplot.plot(Years, AnnualMeanShadow, 'bo', Years, AnnualMeanShadow, 'k')
pyplot.show()


########################  Table (38*4)  ###########################

# Year|| 
# of observ ||
# Mean of biom. this year || 
# Mean of shadow this year  ||

####################################################################

NObserv = [0] * NYears 

for i in range(NYears):
    NObserv[i] = numpy.size(value[year == Years[i]])

dataT3 = {'Year':Years,'Number of Observations':NObserv, 
'mean of biomass':AnnualMeanBiomass,'mean of shadow':AnnualMeanShadow}
df3 = pandas.DataFrame(dataT3)
print(df3)

df3.to_csv(path + 'Table_3.csv', index=False)


##################################################################
#####  Distribution of patches observed in 2007: ###############
##################################################################

Patches2007 = patch[year == 2007]
Biomass2007 = biomass[year == 2007]
Shadow2007 = shadow[year == 2007]

pyplot.hist(Biomass2007, bins = 100)
pyplot.show()

pyplot.hist(Shadow2007, bins = 100)
pyplot.show()

##################################################################
#####  Simulate Random Walk 12 years ahead from 2007
# for some patch which was observed in 2007: ###############
##################################################################
mu = 0.021   # mean   numpy.mean(DeltasYearAvBiomass)
sigma = 0.512 # standard deviation   numpy.std(DeltasYearAvBiomass)

Iteartions = 841  # the number of simulations 
# numpy.size(Biomass2007) = 841

# for ex., 1000 simulations of 12-year forward predictions:
# select randomly patch which was observed in 2007
 # (Y0,Y2, ... , Y11) ... (Y0,Y2, ... , Y11) - 1000 arrays like this
 
Simulations = [[0] * 12 for i in range(Iteartions)] 

for i in range(Iteartions):
    dat1 = data[(year == 2007) & (patch == random.choice (Patches2007))]
    value = dat1[0][2] 
    Simulations[i][0] = value + numpy.random.normal(mu, sigma) # 2008 prediction
#   Switch between these two:
#   value = dat1[0][2] # biomass on the chosen patch in 2007
#   value = dat1[0][3] # shadow on the chosen patch in 2007


for j in range(11):
    for i in range(Iteartions):
        Simulations[i][j+1] = Simulations[i][j] + numpy.random.normal(mu, sigma) 
    


###################################################################
######################### Histograms - predictions: #####################
####################################################################           

path1 = '/Users/Olga Rumyantseva/Desktop/visuals/'


fig = pyplot.hist(Biomass2007, bins = 100, color = 'forestgreen')
matplotlib.pyplot.savefig(path1 + 'BioHist_2007.png')

#fig = pyplot.hist(Shadow2007, bins = 100, color = 'darkolivegreen')
#matplotlib.pyplot.savefig(path1 + 'ShadHist_2007.png')


Predictions2019 = [0] * Iteartions 
for i in range(Iteartions):
    Predictions2019[i] = Simulations[i][11]
    

pyplot.hist(Biomass2007, bins = 100, label='biomass in 2007', color = 'forestgreen')
pyplot.hist(Predictions2019, bins = 100, label='biomass in 2019', color = 'lime')
#pyplot.hist(Shadow2007, bins = 100, label='basal area in 2007', color = 'darkolivegreen')
#pyplot.hist(Predictions2019, bins = 100, label='basal area in 2019', color = 'darkkhaki')
pyplot.legend(loc='upper right')
pyplot.show()
