
'''
# coding: utf-8
# Creates the python dictionary of pandas Dataframes from the .csv or .json files. 
# The output is stored in a pickle data file. 
# written by Nicola Pastorello 24/11/2015

'''


''' Libraries '''
from scripts import *

inputpath = 'Fileset0_json'
dicDF = json2df(inputpath, overwrite=False)

if dicDF: print "DONE!"