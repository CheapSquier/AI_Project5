import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import math
import os
import argparse
from enum import Enum
import matplotlib.pyplot as plt
import time
import warnings
import support_code.ml_file_io as fileIO
import support_code.ml_model_ann as ann
import support_code.ml_model_dtree as dtree

warnings.simplefilter(action='ignore', category=FutureWarning)

def normalizeInputs(inputCol):
    print("*** In normalizeInput, not confident we're correctly normalizing. ***")
    # Should we be modifying in place or returning a copy?
    #WIP: return normalize(inputCol)
    return

_Debug = True

parser = argparse.ArgumentParser(description="Reads a csv data file and a JSON control file to perform analysis with.")
parser.add_argument('datafile', type=str, help="Specify the data file to analyze. Must be csv.")
parser.add_argument('ctrlfile', type=str, help="Specify the analysis control file. Must be JSON.")
parser.add_argument('-file_output', type=str, default='analysis_solution.txt', help="Where to save the analysis output. (default: 'analysis_solution.json')")
args = parser.parse_args()

# Read the Control File to get data file parameters and model parameters
ctrlFileParams = fileIO.getCtrlFileParams(args.ctrlfile)

# Read the data file.
dataDF = pd.read_csv(args.datafile, 
#            usecols = ["ID", "target", "x", "y"],
            usecols = ctrlFileParams["dcols"],
            #dtype = {"ID":str, "target":int, "x":float, "y":float},
            dtype = dict(zip(ctrlFileParams["dcols"], ctrlFileParams["dtypes"])),
            index_col = ctrlFileParams["index"])
# If we need to add a column: dataDF['PFstore'] = True #Adds a column to store PF results

# Do that partition here, get a train and a validation Index
random_seed = 42
trainIdx, validIdx = train_test_split(dataDF.index, test_size = .3, random_state = random_seed)

# Now we have the data and all the model parameters, run the training and validation, check results.
# Might need to massage the data first before actually running the model (nomralize, discretize, partition, etc)

if ctrlFileParams["modelType"] == "ann":

    if ctrlFileParams["normalizeInputs"] == "yes":
        for input in ctrlFileParams["inputcols"]:
            normalizeInputs(dataDF[input])

    # Instantiate the NNet
    NNet = ann.mlp_nn(len(ctrlFileParams["inputcols"]),
                    ctrlFileParams["modelParams"]["numHiddenLyrs"],
                    ctrlFileParams["modelParams"]["numHiddenNodesPerLyr"],
                    len(ctrlFileParams["target"]),
                    ctrlFileParams["modelParams"]["activation"],
                    ctrlFileParams["modelParams"]["learningRate"],
                    ctrlFileParams["startWeight"])

    if _Debug: NNet.printShapes()

    ann.train_ANN(NNet, ctrlFileParams, trainIdx, dataDF)

    # Validate the model
    ann.validate_ANN(NNet, ctrlFileParams, validIdx, dataDF)

if ctrlFileParams["modelType"] == "dtree":

    if ctrlFileParams["discretizeInputs"] == "yes":
        for input in ctrlFileParams["inputcols"]:
            normalizeInputs(dataDF[input])

    #Since we do column query operations during training to calculate entropy, we're going actually split
    #DataDF into TrainDF and ValidDF based on the trainIdx and validIdx (instead of just iterating over
    #those Idx values to calculate each record).
    trainDF = dataDF.loc[trainIdx, dataDF.columns]
    validDF = dataDF.loc[validIdx, dataDF.columns]

    # Instantiate the Tree
    DTree = dtree.dtree_model(len(ctrlFileParams["inputcols"]),
                    len(ctrlFileParams["target"]),
                    ctrlFileParams["modelParams"]["splitOn"],
                    ctrlFileParams["modelParams"]["maxDepth"])
    # ================  Training  ================
    nodeID = 0
    DFDict ={}
    layer = 0
    # Process the Root Node in Layer 0
    splitVals = dtree.getSplitCriteria(DTree, ctrlFileParams, trainDF)
    DTree.nodeDict[nodeID] = dtree.dtree_node(nodeID, 
                                        None,               # Parent Node ID
                                        min(splitVals, key=splitVals.get),    # feature to split on
                                        len(trainDF[min(splitVals)].unique())) # Number of splits
    DFDict[nodeID] = trainDF # Keep separate from dtree object. Won't be needed after training.
    branchValues = trainDF[DTree.nodeDict[nodeID].splitVariable].unique()
    print("Layer {} will split on {}".format(layer, DTree.nodeDict[nodeID].splitVariable))
    #Pre-processing Layer 1
    DTree.layer[layer+1] = []
    for newNode in range(DTree.nodeDict[nodeID].numChildren):
        nextNodeID = nodeID+newNode+1
        DTree.layer[layer+1].append(nextNodeID)
        DTree.nodeDict[nextNodeID] = dtree.dtree_node(nextNodeID, nodeID, None, None, branchValues[newNode])
        DTree.nodeDict[nodeID].addChildNode(nextNodeID)
        DTree.nodeDict[nextNodeID].valueDict = DTree.nodeDict[nodeID].valueDict.copy() #Start with parent values
        DTree.nodeDict[nextNodeID].valueDict[DTree.nodeDict[nodeID].splitVariable] = branchValues[newNode]
        DFDict[nextNodeID] = DFDict[nodeID].query(dtree.makeTreeQuery(DTree.nodeDict[nextNodeID].valueDict))
    #Now continue for the other layers
    for layer in range(1, ctrlFileParams["modelParams"]["maxDepth"]):
        print("Find split(s) for layer:", layer)
        for nodeID in DTree.layer[layer]:
            print("Evaluating node ", nodeID)
            nodeDF = DFDict[nodeID]
            splitVals = dtree.getSplitCriteria(DTree, ctrlFileParams, nodeDF)
            if splitVals == {}:
                DTree.nodeDict[nodeID].leaf = True
                print("Layer {}, node {} is a leaf node".format(layer, nodeID))
                continue
            #First, update this node
            DTree.nodeDict[nodeID].splitVariable = min(splitVals, key=splitVals.get) #Still not sure how this works...
            DTree.nodeDict[nodeID].numChildren = len(nodeDF[DTree.nodeDict[nodeID].splitVariable].unique())
            branchValues = nodeDF[DTree.nodeDict[nodeID].splitVariable].unique()
            print("Layer {}, node {} will split on {}".format(layer, nodeID, DTree.nodeDict[nodeID].splitVariable))
        #After finding the splits, pre-processing next layer
        DTree.layer[layer+1] = [] #next layer
        for nodeID in DTree.layer[layer]: #current layer
            if DTree.nodeDict[nodeID].leaf == True:
                continue
            for newNode in range(DTree.nodeDict[nodeID].numChildren):
                nextNodeID = len(DTree.nodeDict) #nodeID+newNode+1
                DTree.layer[layer+1].append(nextNodeID)
                DTree.nodeDict[nextNodeID] = dtree.dtree_node(nextNodeID, nodeID, None, None, branchValues[newNode])
                DTree.nodeDict[nodeID].addChildNode(nextNodeID)
                DTree.nodeDict[nextNodeID].valueDict = DTree.nodeDict[nodeID].valueDict.copy() #Start with parent values
                DTree.nodeDict[nextNodeID].valueDict[DTree.nodeDict[nodeID].splitVariable] = branchValues[newNode]
                DFDict[nextNodeID] = DFDict[nodeID].query(dtree.makeTreeQuery(DTree.nodeDict[nextNodeID].valueDict))
    # Assign output values to leaf nodes based on majority of target records
    for node in DTree.nodeDict:
        if DTree.nodeDict[node].leaf == False:
            continue
        #Each leaf node has its own subset dictionary. Get majority target values from that.
        DTree.nodeDict[node].result = DFDict[node][ctrlFileParams["target"]].mode().at[0] # mode returns Series, so take 1st value

    # First get error rate on the training set
    trainingErr = dtree.run_dtree(DTree, ctrlFileParams, trainDF)

    # Validate the model
    validationErr = dtree.run_dtree(DTree, ctrlFileParams, validDF)

    DTree.printTreeInfo()
    print("Training Error: ", trainingErr)
    print("Validation Error: ", validationErr)


a=1
