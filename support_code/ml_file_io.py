import json
import pandas

"""
# Let pandas determine the column names, but we'll have to create a list of them to 
# iterate on.
REALfullDataDF = pd.read_csv('train-file.csv', 
#            usecols = ["ID", "target", "x", "y"],
#            dtype = {"ID":str, "target":int, "x":float, "y":float},
            index_col = "ID")
REALfullDataDF['PFstore'] = True #Adds a column to store PF results
"""

def getCtrlFileParams(controlfile):
    #Returns dict of input & output information, and model info
    try:
        if controlfile[-5:] != ".json": print("Error: control file isn't JSON")
        f = open(controlfile, "rt")
        ctrlParams = json.load(f)
    except:
        print("Error: Control file not found, or JSON syntax problem: ", controlfile)
    #expected keys: dtypes:[list of np.<type>], target:<target col name>, normalize:[list of inputs to normalize],
    #                       discretize:[list of inputs to discretize], discParams:[list of values, TBD],
    #                       modelType:[knn|ann|dt], modelParams:[list of values, TBD]
    #Assumes row 0 is the col name header
    #Assumes col 0 is record ID
    #target,x,y,PFstore
    return ctrlParams
