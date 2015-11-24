'''
# coding: utf-8
# Various functions and objects
# written by Nicola Pastorello 24/10/2015
# 
'''

''' Libraries '''
import glob, json, os, pickle, sys
import numpy as np
import pandas as pd

''' Functions '''

def csv2df(inputPath, overwrite=False): 
    ''' Loads the .csv files and returns a dictionary of 
        pandas dataframes '''
    # Reading csv files
    if os.path.exists('dicDF.dat') and not(overwrite):
        fileIn = open('dicDF.dat', 'rb')
        dicDF = pickle.load(fileIn)
        fileIn.close()
        return dicDF

    # Creating pandas dataframe
    dicDF = {}
    listFail = []
    for ii in glob.glob(inputPath+'/*.csv'):
        try:
            df = pd.read_csv(ii)
            # Removing zeropoint
            tmp = pd.to_datetime(df['time'])-pd.to_datetime(df['time'][0])
            df['time'] = tmp
            dicDF[ii] = df
        except:
            ee = sys.exc_info()[0]
            listFail.append([ii, ee])

    # Check exceptions
    if len(listFail) > 0: 
        print "Errors with the following files:"
        for ii in listFail:
            print "\n"+ii[0]
        return False
    else:
        print "No errors"
        # Saving output
        fileOut = open('dicDF.dat', 'wb')
        pickle.dump(dicDF, fileOut)
        fileOut.close()
        return dicDF

def json2df(inputPath, overwrite=False): 
    ''' Loads the .json files and returns a dictionary of 
        pandas dataframes '''

    # Reading csv files
    if os.path.exists('dicDF.dat') and not(overwrite):
        fileIn = open('dicDF.dat', 'rb')
        dicDF = pickle.load(fileIn)
        fileIn.close()
        return dicDF

    # Creating pandas dataframe
    dicDF = {}
    listFail = []
    for ii in glob.glob(inputPath+'/*.json'):
        try:
            df = pd.read_json(ii)
            # Removing zeropoint
            tmp = pd.to_datetime(df.index)-pd.to_datetime(df.index[0])
            df['time'] = tmp
            dicDF[ii] = df
        except:
            ee = sys.exc_info()[0]
            listFail.append([ii, ee])

    # Check exceptions
    if len(listFail) > 0: 
        print "Errors with the following files:"
        for ii in listFail:
            print "\n"+ii[0]
        return False
    else:
        print "No errors"
        # Saving output
        fileOut = open('dicDF.dat', 'wb')
        pickle.dump(dicDF, fileOut)
        fileOut.close()
        return dicDF
