
'''
# coding: utf-8
# Extracting the data and converting it in a JSON object 
# written by Nicola Pastorello 24/10/2015

# The code reads the input data (in .txt format) and converts it into .json and .csv files. 
'''

''' Libraries '''
import glob, json, os, sys
import numpy as np

''' Functions '''

def findCharPos(stringInput, c=' '): #Finds position of a character in a string
    for ii in np.arange(len(stringInput)):
        if stringInput[ii] == c:
            return ii, stringInput[:ii]
    return False

def removeFChar(stringInput, cc): #Remove extra characters at the end of a string
    for ii in np.arange(len(stringInput)-1, -1, -1):
        if stringInput[ii] != cc:
            return stringInput[:ii+1]



''' Main '''
# Loading data 
listFiles = glob.glob('../Fileset0/*.txt')


if not(os.path.exists('Fileset0_json')):
    os.mkdir('Fileset0_json')

if not(os.path.exists('Fileset0_csv')):
    os.mkdir('Fileset0_csv')

listFail = []
for ii in listFiles:
    try:
        fileIn = open(ii, 'rb'); f = fileIn.read(); fileIn.close()
        ID = findCharPos(f, c='\t')[1]

        # Position of first row
        posFirstRow = findCharPos(f, c=':')[0]-2
            
        # Reading line by line and storing data in a list for the csv
        listRows, listTmp = ['time,hb,accel,capacitorDisconnect'], []
        for jj in np.arange(len(f)-posFirstRow):
            if f[posFirstRow+jj] == '\t':
                listTmp.append(',')
            elif (f[posFirstRow+jj] != '\r') and (f[posFirstRow+jj] != '\n'):
                listTmp.append(f[posFirstRow+jj])
            else:
                totLine = ''.join(listTmp)
                if totLine:
                    listRows.append(removeFChar(totLine, ','))
                    listTmp = [] 
        
        outTab = '\n'.join(listRows)

        fileOut = open('Fileset0_csv/'+ii[12:-4]+'.csv', 'wb')
        fileOut.write(outTab)
        fileOut.close()

        tmp_DF = pd.DataFrame.from_csv('Fileset0_csv/'+ii[12:-4]+'.csv', sep=',')
        tmp_DF.to_json('Fileset0_json/'+ii[12:-4]+'.json')

    except:
        ee = sys.exc_info()[0]
        listFail.append([ii, ee])

# Check exceptions
if len(listFail) > 0: 
    print "Errors with the following files:"
    for ii in listFail:
        print "\n"+ii[0]
else:
    print "No errors"