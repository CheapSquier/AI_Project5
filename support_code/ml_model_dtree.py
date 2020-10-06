import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import math
import os

class dtree_node:

    def __init__(self, nodeID, parentNode, splitVar, numChildren, branchVal = None):

        self.nodeID = nodeID
        self.splitVariable = splitVar
        self.branchVal = branchVal #Root node won't have a branchVal
        self.numChildren = numChildren
        self.parentNodeID = parentNode
        self.childNodeIDs = []
        self.valueDict = {}
        self.nodeDF = None
        if numChildren == 0:
            self.leaf = True
        else: self.leaf = False
        self.result = None

    def addChildNode(self, nodeID):
        self.childNodeIDs.append(nodeID)

class dtree_model:

    def __init__(self, numInputs, numOutputs, splitOn, maxDepth):

        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.splitOn = splitOn
        self.maxDepth = maxDepth
        self.layer = {}
        self.layer[0] = [0] #Will be a dict of layers containing lists of nodes. Layer 0 and root node always the same
        self.nodeDict = {} #this will be a list of node objects, each with their own data columns and splits

    def printTreeInfo(self):
        print("{:10} {:10} {:10} {:10} {:10}".format("Node","Parent","Children","SplitOn","values"))
        for node in self.nodeDict:
            print(" {:10} {:10} {:10} {:10} {}".format(str(node), 
                                                        str(self.nodeDict[node].parentNodeID), 
                                                        str(self.nodeDict[node].childNodeIDs), 
                                                        str(self.nodeDict[node].splitVariable), 
                                                        str(self.nodeDict[node].valueDict)))                                                 
        return

def getEntropy(valuesTarget, valuesInput,  valueArray, numRecords):
    entropy = 0
    for inValue in range(len(valuesInput)):
        tempSum = 0
        numVals = 0
        for outValue in range(len(valuesTarget)):
            numVals += valueArray[outValue, inValue]
            prob = valueArray[outValue, inValue] / valueArray[:,inValue].sum()
            if prob == 0:
                tempSum -= 0
            else: tempSum -= (prob*math.log2(prob))
        entropy += tempSum*(numVals/numRecords)
    return entropy

def getGini(valuesTarget, valuesInput,  valueArray, numRecords):
    gini = 0
    for inValue in range(len(valuesInput)):
        tempSum = 0
        numVals = 0
        for outValue in range(len(valuesTarget)):
            numVals += valueArray[outValue, inValue]
            prob = valueArray[outValue, inValue] / valueArray[:,inValue].sum()
            if prob == 0:
                tempSum -= 0
            else: tempSum -= (prob*math.log2(prob))
        gini += tempSum*(numVals/numRecords)
    return gini

def getSplitCriteria(DTree, ctrlFileParams, dataDF):
    # Get Entropy/Gini for each input variable (column)
    valueTables = {} # index on inputCols
    splitVal = {}
    for inputCol in ctrlFileParams["inputcols"]:
        valuesTarget = dataDF[ctrlFileParams["target"]].unique() #.unique returns an array of all unqiue values
        if len(valuesTarget) ==1:
            #only one value here, this is a leaf node
            return splitVal
        valuesInput = dataDF[inputCol].unique()
        valueTables[inputCol] = np.empty([len(valuesTarget), len(valuesInput)]) # Make an array to store values
        outIdx = -1
        for outValue in valuesTarget:
            outIdx += 1
            inIdx = -1
            for inValue in valuesInput: # We're training with the trainDF so if a value is in validDF but not trainDF, that could Err
                inIdx += 1
                outQuery =str(ctrlFileParams["target"] + " == " + str(outValue))
                inQuery = str(inputCol + " == " + str(inValue))
                valueTables[inputCol][outIdx, inIdx] = dataDF.query(outQuery).query(inQuery)[inputCol].count()
        if DTree.splitOn == "entropy":
            splitVal[inputCol] = getEntropy(valuesTarget, valuesInput,  valueTables[inputCol], len(dataDF.index))
        if DTree.splitOn == "gini":
            splitVal[inputCol] = getGini(valuesTarget, valuesInput,  valueTables[inputCol], len(dataDF.index))

    return splitVal

def makeTreeQuery(valueDict):
    myQuery = ""
    term = len(valueDict)
    for setting in valueDict:
        term -= 1
        myQuery += setting + "==" + str(valueDict[setting])
        if term != 0:
            myQuery += " and "
    return myQuery

def run_dtree(DTree, ctrlFileParams, theDF):
    # test data with the tree and calculate an error
    correctNum = 0
    for node in DTree.nodeDict:
        if DTree.nodeDict[node].leaf == False:
            continue
        tempDF = theDF.query(makeTreeQuery(DTree.nodeDict[node].valueDict))
        correctNum += len(tempDF.query(ctrlFileParams["target"] + "==" + str(DTree.nodeDict[node].result)).index)
    totalErr = correctNum/len(theDF.index)

    return totalErr