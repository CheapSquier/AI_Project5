import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import math
import os

def normalizeInputs(inputCol):
    print("*** In normalizeInput, not confident we're correctly normalizing. ***")
    # Should we be modifying in place or returning a copy?
    return normalize(inputCol)

def partitionData(df):
    #Maybe just do this in ml_main
    return trainDF, validDF

def train_ANN(NNet, ctrlFileParams):

    for epoch in range(ctrlFileParams["modelParams"]["epochs"]):
        # Train the model (remember, rows x columns)
        totalTrainErr = 0
        for inData in trainIdx:
            # Set input layer to input values for this index in the data.
            NNet.inLayer = dataDF.loc[inData, ctrlFileParams["inputcols"]]
            #Now forward propagation
                #First hidden layer is a little different since using input weights, not previous hidden layer
                #interating would be more elegant but require __iter__ defn and maybe more class defs
            NNet.hidLayersIn[0] =  NNet.inLayer.T.dot(NNet.inWeights) # Sum of the weights and the inputs
            for node in range(0, NNet.nodesPerHidden):
                NNet.hidLayers[0][node] = NNet.activResult(NNet.hidLayersIn[0][node]) #[0] selects a list member (1st hid Layer) which returns an ndarray
                                                                        #[node] returns a subset of that ndarray
            if NNet.numHiddenLyrs > 1:
                for layer in range(1, NNet.numHiddenLyrs): 
                    NNet.hidLayersIn[layer] =  NNet.hidLayers[layer - 1].T.dot(NNet.hidWeights[layer - 1])
                    for node in range(0, NNet.nodesPerHidden):
                        NNet.hidLayers[layer][node] = NNet.activResult(NNet.hidLayersIn[layer][node])
            NNet.outLayerIn =  NNet.hidLayers[NNet.numHiddenLyrs - 1].T.dot(NNet.outWeights) # Sum of the weights and the inputs
            for node in range(NNet.numOutputs):
                NNet.outLayer[node] = NNet.activResult(NNet.outLayerIn[node])
                totalTrainErr += abs(NNet.outLayer[node] - dataDF['target'].loc[inData])
            #Now back propagation
            #in_j, y_j_target, a_j
            NNet.backPropagate(NNet.outDeltas, NNet.outLayerIn, dataDF['target'].loc[inData], NNet.outLayer)
            #in_i, weight_ij, delta_j
            delta_j = NNet.outDeltas
            if NNet.numHiddenLyrs > 1:
                for layer in range(NNet.numHiddenLyrs - 1, 0, -1):
                    NNet.hidDeltas[layer] = NNet.backPropagateHid(NNet.hidDeltas[layer], NNet.hidLayersIn[layer], NNet.hidWeights[layer-1], delta_j)
                    delta_j = NNet.hidDeltas[layer]
            #else:
            #    NNet.hidDeltas[layer] = NNet.backPropagateHid(NNet.hidDeltas[layer], NNet.hidLayersIn[layer], NNet.hidWeights[layer-1], delta_j)
            #update the weights
            for node in range(NNet.numOutputs):
                NNet.outWeights[node] = NNet.outWeights[node] + NNet.learnRate*NNet.hidLayers[-1][node]*NNet.outDeltas[node]
            for layer in range(NNet.numHiddenLyrs - 2, 0 - 1, -1): 
                """
                for node in range(NNet.nodesPerHidden):
                    NNet.hidWeights[layer][node] = NNet.hidWeights[layer][node] + NNet.learnRate*NNet.hidLayers[layer][node]*NNet.hidDeltas[layer+1][node]
                """
                for node_i in range(NNet.nodesPerHidden):
                    for node_j in range(NNet.nodesPerHidden):
                        NNet.hidWeights[layer][node_i,node_j] = NNet.hidWeights[layer][node_i,node_j] + NNet.learnRate*NNet.hidLayers[layer][node_j]*NNet.hidDeltas[layer+1][node_j]
            for nodeIn in range(NNet.numInputs):
                for nodeHid in range(NNet.nodesPerHidden):
                    NNet.inWeights[nodeIn][nodeHid] = NNet.inWeights[nodeIn][nodeHid] + NNet.learnRate*NNet.inLayer[nodeIn]*NNet.hidDeltas[0][nodeHid]
            
        print("Epoch: {}, Training Err: {}".format(epoch, totalTrainErr))

    return

def validate_ANN(NNet, ctrlFileParams):
    # Validate the model
    totalValidErr = 0
    for inData in validIdx:
        # Set input layer to input values for this index in the data.
        NNet.inLayer = dataDF.loc[inData, ctrlFileParams["inputcols"]]
        #Now forward propagation
            #First hidden layer is a little different since using input weights, not previous hidden layer
            #interating would be more elegant but require __iter__ defn and maybe more class defs
        NNet.hidLayersIn[0] =  NNet.inLayer.T.dot(NNet.inWeights) # Sum of the weights and the inputs
        for node in range(0, NNet.nodesPerHidden):
            NNet.hidLayers[0][node] = NNet.activResult(NNet.hidLayersIn[0][node]) #[0] selects a list member (1st hid Layer) which returns an ndarray
                                                                       #[node] returns a subset of that ndarray
        if NNet.numHiddenLyrs > 1:
            for layer in range(1, NNet.numHiddenLyrs): 
                NNet.hidLayersIn[layer] =  NNet.hidLayers[layer - 1].T.dot(NNet.hidWeights[layer - 1])
                for node in range(0, NNet.nodesPerHidden):
                    NNet.hidLayers[layer][node] = NNet.activResult(NNet.hidLayersIn[layer][node])
        NNet.outLayerIn =  NNet.hidLayers[NNet.numHiddenLyrs - 1].T.dot(NNet.outWeights) # Sum of the weights and the inputs
        for node in range(NNet.numOutputs):
            NNet.outLayer[node] = NNet.activResult(NNet.outLayerIn[node])
        totalValidErr += abs(NNet.outLayer[node] - dataDF['target'].loc[inData])
        #print("Actual: {}, Expected: {}".format(NNet.outLayer[node], dataDF['target'].loc[inData][0]))

    print("Total Validation Error:", totalValidErr)
    return