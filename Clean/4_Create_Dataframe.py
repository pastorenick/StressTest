
# coding: utf-8

# ### Collect the various outputs and build up a total dataframe

# ###### written by Nicola Pastorello 30/10/2015

# In[4]:

get_ipython().magic(u'matplotlib inline')
# Loading libraries
import numpy as np
import pandas as pd
import datetime, pickle
import os, glob, sys
from pylab import *
from scipy import signal, misc, optimize


# In[8]:

# Retrieve data
fileIn = open('dicDF.dat', 'rb')
dicDF = pickle.load(fileIn)
fileIn.close()


fileIn = open('fitData.dat', 'rb')   #DATA FITTED WITH THREE LINES FROM ipython notebook
dicFit = pickle.load(fileIn)
fileIn.close()


# In[13]:

# Retrieve fitted parameters
['Slope', 'SleepTime', 'Chi2', 'AwakeTime', 'Intercept', 'ID']

slope, intercept, chi2, ID_fit, sleepT, awakeT = [], [], [], [], [], []

for ii in np.arange(len(dicFit['AwakeTime'])):    
    ID_fit.append(dicFit['ID'][ii])
    slope.append(dicFit['Slope'][ii])
    intercept.append(dicFit['Intercept'][ii])
    chi2.append(dicFit['Chi2'][ii])
    sleepT.append(dicFit['SleepTime'][ii])
    awakeT.append(dicFit['AwakeTime'][ii])
    

Series_IDfit = pd.Series(ID_fit)
Series_slope = pd.Series(slope)
Series_intercept = pd.Series(intercept)
Series_chi2 = pd.Series(chi2)
Series_sleepT = pd.Series(sleepT)
Series_awakeT = pd.Series(awakeT)


# In[14]:

# Retrieve original profiles data
ID, time, HB, acc, disconnections = [], [], [], [], []
for ii in dicDF.keys():
    ID.append(ii[13:-4])
    time.append(np.array(dicDF[ii]['time']))
    HB.append(np.array(dicDF[ii]['hb']))
    acc.append(np.array(dicDF[ii]['var1']))
    disconnections.append(np.array(dicDF[ii]['var2']))

Series_ID = pd.Series(ID)
Series_HB = pd.Series(np.transpose(HB), dtype=np.dtype("object"))
Series_time = pd.Series(time, dtype=np.dtype("object"))
Series_acc = pd.Series(acc, dtype=np.dtype("object"))
Series_disconnections = pd.Series(disconnections, dtype=np.dtype("object"))


# In[15]:

# Build single dataframe

dic_final = {'ID':Series_ID, 'HB':Series_HB, 'time':Series_time, 
             'acc':Series_acc, 'disconnections':Series_disconnections, 
             'Slope': Series_slope, 'intercept':Series_intercept, 'chi2': Series_chi2, 
             'Sleep': Series_sleepT, 'Awake': Series_awakeT, }

df_final = pd.DataFrame(dic_final) 


# In[16]:

# Adding labels
labelDF = pd.read_csv('FileSet0.stressRatings.csv')
labelDF.columns = ['HLcode', 'Stress']
# Inner join
mergedDF = pd.merge(left=df_final, right=labelDF, 
                  left_on='ID', right_on='HLcode', how='left')


# In[17]:

# Saving
fileOut = open('totDF.dat', 'wb')
pickle.dump(mergedDF, fileOut)
fileOut.close()

# Save DF as as .csv
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
write_csv = robjects.r('write.csv')
write_csv(mergedDF,'totDF.csv')

