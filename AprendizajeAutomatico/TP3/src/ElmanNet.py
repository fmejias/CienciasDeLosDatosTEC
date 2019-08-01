

#Import necessary packages, torch, numpy, pylab
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init

#type for tensors
dtype = torch.FloatTensor
#context and input layer concatenated size, hidden layer dimension
contextConcatInputLayerSize, hiddenLayerSize, outputLayerSize = 7, 6, 1
#number of epochs for training
epochs = 300
#input sequence length
sequenceLength = 20
#model learning rate
learningRate = 0.1

"""
Creates the input and ground truth data
@param sequenceLength, sequence length to generate
@return yInput and yTarget
"""
def createInputAndGroundTruthData(sequenceLength):
    #create data steps from 2 to 10 with the given sequence length
    xTimeSteps = np.linspace(2, 10, sequenceLength + 1)
    #create numpy array with sin(x) input
    yNp = np.sin(xTimeSteps)
    yNp.resize((sequenceLength + 1, 1))
    #create the input time series for the model, with one unit of delay, is no model parameter, no grad needed
    yInput = Variable(torch.Tensor(yNp[:-1]).type(dtype), requires_grad = False)
    # create the target or ground truth data
    yTarget = Variable(torch.Tensor(yNp[1:]).type(dtype), requires_grad = False)
    return (xTimeSteps[1:], yInput, yTarget)

"""
Creates the matrices for the Elman model, in this case W1 and V
@param contextConcatInputLayerSize
@param hiddenLayerSize
@param outputLayerSize
@return W1 and V parameter matrices
"""
def createElmanModelParameters(contextConcatInputLayerSize, hiddenLayerSize, outputLayerSize):
    #W1 with contextConcatInputLayerSize x hiddenLayerSize dimensions
    W1 = torch.FloatTensor(contextConcatInputLayerSize, hiddenLayerSize).type(dtype)
    #randomly init W1 parameter matrix with mean 0 and std 0.4
    init.normal(W1, 0.0, 0.4)
    #pytorch variable with grad requirement
    W1 = Variable(W1, requires_grad = True)
    # randomly init V parameter matrix with mean 0 and std 0.3
    V = torch.FloatTensor(hiddenLayerSize, outputLayerSize).type(dtype)
    init.normal(V, 0.0, 0.3)
    # pytorch variable with grad requirement
    V = Variable(V, requires_grad = True)
    return (W1, V)

"""
Model forward pass
@param input, current input in t
@param contextState, previous output in (t - 1) the sequence of hidden states
@param W1
@param V
"""
def forward(input, contextState, W1, V):
  #concatenate input and context state
  xAndContext = torch.cat((input, contextState), 1)
  #calculate next context state (hidden output for current t) with tanh(xAndContext * W1)
  contextState = torch.tanh(xAndContext.mm(W1))
  #output = h * V, with identity activation function
  output = contextState.mm(V)
  return  (output, contextState)


"""
Trains the model with an squared error loss function
"""
def trainModel(learningRate, epochs,  hiddenLayerSize, x, y,  W1, V):
    #for each epoch...
    for i in range(epochs):
        #total model loss
        totalLoss = 0
        #init the array of context state units
        contextState = Variable(torch.zeros((1, hiddenLayerSize)).type(dtype), requires_grad = True)
        #for each time unit in the sequence
        for t in range(x.size(0)):
            #current input in the sequence at t
            input = x[t:(t + 1)]
            #input = x[t] not like that to create a matrix and not a vector
            #current target value at t
            target = y[t];
            #forward pass outputs prediction at time t, and the new context state for t + 1
            (prediction, contextState) = forward(input, contextState, W1, V)
            #calculates the loss for time t
            loss = (prediction - target).pow(2).sum() / 2
            #accumulate loss
            totalLoss += loss
            #backpropagation from the loss
            loss.backward()
            #use gradient of W1 to update it
            W1.data -= learningRate * W1.grad.data
            # use gradient of V to update it
            V.data -= learningRate * V.grad.data
            #set gradients to zero to use it in t + 1
            W1.grad.data.zero_()
            V.grad.data.zero_()
            #update context units for time t + 1 in the sequence
           # print("Old context state ", contextState)
            contextState = Variable(contextState.data)
           # print("New context state", contextState)
        if i % 10 == 0:
            print("Epoch: {} loss {}".format(i, totalLoss.data[0]))
    return (W1, V);

"""
Computes prediction with x as  input 
"""
def predict(x, hiddenLayerSize, W1, V):
    #creates array of context units
    contextState = Variable(torch.zeros((1, hiddenLayerSize)).type(dtype), requires_grad = False)
    #python list of predictions
    predictions = []
    #make prediction for each time unit t
    for t in range(x.size(0)):
        #to create a 2d tensor instead of a 1d tensor
        input = x[t : t + 1]
        #compute prediction and new contextState
        (prediction, contextState) = forward(input, contextState, W1, V)
        contextState = contextState
        #append new prediction, using ravel to return a contiguos flattened array
        predictions.append(prediction.data.numpy().ravel()[0])

    return predictions

"""
Main function
"""
def main():
    #create xTimeSteps, x axis for the function, yInput is the series delayed by t-1, and yTarget
    #the series of values tu estimate
    (xTimeSteps, yInput, yTarget) = createInputAndGroundTruthData(sequenceLength);
    #print(yInput.size())
    #print(xTimeSteps.size)
    #print(yTarget.size())
    #Create Elmans parameters
    (W1, V) = createElmanModelParameters(contextConcatInputLayerSize, hiddenLayerSize, outputLayerSize)
    #Estimate Elmans parameters
    (W1, V) = trainModel(learningRate, epochs, hiddenLayerSize, yInput, yTarget, W1, V)
    yPredicted = predict(yInput, hiddenLayerSize, W1, V)
    #Plot groundtruth
    plt.figure()
    plt.scatter(xTimeSteps, yTarget.numpy())
    plt.show()
    #Plot estimated series
    plt.figure()
    plt.scatter(xTimeSteps, yPredicted)
    plt.show()
main()