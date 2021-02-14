#
#
#this file is a cleaned up version of the file SNNMOD2.py
#Test change 1
#
#

#this is a file for designing the SNN using CUDA and a more optimal construction
#this is based on the code described in https://developer.nvidia.com/blog/numba-python-cuda-acceleration/

#and this code as well https://nyu-cds.github.io/python-numba/05-cuda/

#runtime from: https://www.geeksforgeeks.org/python-measure-time-taken-by-program-to-execute/


import numpy as np
from numba import cuda
#from numba import *
import random

#first we are probably going to want to start with a set of matrices
#we are going to want several matrices for each layer, with each index representing a neuron
#we will need one for all of the weights
#we will need one for all of the inhibitions/refactory periods
#we will need one for all of the potentials
#we might also want some for homeostasis (if we want to keep that)
#and we might want some for leaks(if we want to re add that feature)


class SpikingCudaNetwork:
    def __init__(self, layerSizes, windowSize, refactory, learningRate, learningRate2, leak, threshold, filename="NONE"):
        self.threshold = threshold
        self.learningRate = learningRate
        self.learningRate2 = learningRate2 
        self.windowSize = windowSize
        self.layerSizes = layerSizes
        #the output tracker for the output layer
        self.outputTimes = cuda.to_device(np.full(layerSizes[-1], -1))

        #layer sizes is a set of integers, so we make a matrix for each of the layers of that size
        self.layersPotential = []

        #0 if no spike, 1 if spike, -1 if can't spike
        self.spikes = []
        #creates all of the layers starting potential
        for i in self.layerSizes:
            self.layersPotential.append(cuda.to_device(np.zeros(i)))
            self.spikes.append(cuda.to_device(np.zeros(i)))

        #Spiking times for each neuron
        self.spikeTime = []
        #creates all of the layers starting potential
        for i in self.layerSizes:
            self.spikeTime.append(cuda.to_device(np.full(i, -1)))


        #creating the weights arrays
        self.weightsTemp = []
        self.weightsTemp.append(np.ones((1, 1)))
        for i in range(1, len(layerSizes)):
            self.weightsTemp.append(np.ones((layerSizes[i], layerSizes[i - 1])))

        self.createRandomWeights()

        self.weights = []
        for i in range(0, len(self.weightsTemp)):
            self.weights.append(cuda.to_device(self.weightsTemp[i]))

        #this keeps track of the inputs that we received
        self.inputTracker = []
        self.inputTracker.append(cuda.to_device(np.zeros((1, 1))))

        #this keeps track of the times for each input
        self.inputTimes = []
        self.inputTimes.append(cuda.to_device(np.zeros((1, 1))))

        #this keeps track of how many inputs
        self.inputTrackerCount = []
        self.inputTrackerCount.append(-1)

        for i in range(1, len(layerSizes)):
            self.inputTracker.append(cuda.to_device(np.full((layerSizes[i], layerSizes[i - 1]), -1)))
            self.inputTimes.append(cuda.to_device(np.full((layerSizes[i], layerSizes[i - 1]), -1)))
            self.inputTrackerCount.append(cuda.to_device(np.full((layerSizes[i]), 0)))

    #resets the needed parts of the network in preparation for the next set of inputs
    def reinitialize(self):
        self.outputTimes = cuda.to_device(np.full(self.layerSizes[-1], -1))
        self.layersPotential = []
        self.spikes = []
        #creates all of the layers starting potential
        for i in self.layerSizes:
            self.layersPotential.append(cuda.to_device(np.zeros(i)))
            self.spikes.append(cuda.to_device(np.zeros(i)))

        #Spiking times for each neuron
        self.spikeTime = []
        #creates all of the layers starting potential
        for i in self.layerSizes:
            self.spikeTime.append(cuda.to_device(np.full(i, -1)))

        self.inputTracker = []
        self.inputTracker.append(cuda.to_device(np.zeros((1, 1))))
        self.inputTimes = []
        self.inputTimes.append(cuda.to_device(np.zeros((1, 1))))
        self.inputTrackerCount = []
        self.inputTrackerCount.append(-1)

        for i in range(1, len(self.layerSizes)):
            self.inputTracker.append(cuda.to_device(np.full((self.layerSizes[i], self.layerSizes[i - 1]), -1)))
            self.inputTimes.append(cuda.to_device(np.full((self.layerSizes[i], self.layerSizes[i - 1]), -1)))
            self.inputTrackerCount.append(cuda.to_device(np.full((self.layerSizes[i]), 0)))

    @cuda.jit
    def advanceTimeCUDA(prevSpikes, potential, postSpikes, layerWeights, layerindex, history, historyTimes, historyCount, time, threshold, spikeTime):

        index = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

        if index < potential.size:

            for i in range(0, layerWeights[layerindex].size):

                #skip neurons that have already spiked
                if postSpikes[index] == -1 and prevSpikes[i] == 1:
                    #print("history count")
                    #print(historyCount[index])
                    history[index][historyCount[index]] = i
                    historyTimes[index][historyCount[index]] = time
                    historyCount[index] += 1
                    continue

                #layerWeights[index][i] += 1
                #keeps track of the input spikes
                if prevSpikes[i] == 1:
                    potential[index] += layerWeights[index][i] * prevSpikes[i]
                    history[index][historyCount[index]] = i
                    historyTimes[index][historyCount[index]] = time
                    historyCount[index] += 1
                #determines if this neuron spiked
                if potential[index] >= threshold and postSpikes[index] != -1:
                    postSpikes[index] = 1
                    spikeTime[index] = time

        #TODO might also want to clear prevSpikes here
        #i think that this has to be here, but I am not sure
        cuda.syncthreads()

    @cuda.jit
    def CUDAInput(spikes, inputSpikes, timeStep):
        neuron = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        if neuron < spikes.size:

            if timeStep == inputSpikes[neuron]:
                spikes[neuron] = 1
            else:
                spikes[neuron] = 0
        cuda.syncthreads()

    # spikes is the spikes array of the output layer
    # outputTimes is an array to store the output times at index = neuron index
    # timestep is the current time step
    @cuda.jit
    def CUDAoutputTracker(spikes, outputTimes, timeStep):
        neuron = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        if neuron < spikes.size:
            if spikes[neuron] == 1:
                outputTimes[neuron] = timeStep
        cuda.syncthreads()

    @cuda.jit
    def adjustWeights(layer, weights, spikes, history, historyTimes, historyCount, learningRate, learningRate2, STDP, time, spikeTime):
        neuron = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

        if neuron < spikes.size and spikes[neuron] == 1:
            spikes[neuron] = -1
            if STDP is True:
                for i in range(0, historyCount[neuron]):
                    #weights[neuron][history[neuron][i]] += (historyTimes[neuron][i] / time) * learningRate * weights[neuron][history[neuron][i]] * (1.0 - weights[neuron][history[neuron][i]])
                    deltaT = (historyTimes[neuron][i] - time)
                    deltaT = deltaT / 100.0
                    deltaW = 1 * 2.71828**(deltaT / .02)
                    weights[neuron][history[neuron][i]] += deltaW * learningRate

            historyCount[neuron] = 0

        elif neuron < spikes.size and spikes[neuron] == -1:
            spikes[neuron] = -1

            if STDP is True:
                for i in range(0, historyCount[neuron]):
                    #weights[neuron][history[neuron][i]] -= (learningRate2) * weights[neuron][history[neuron][i]] * (1.0 - weights[neuron][history[neuron][i]])
                    deltaT = (spikeTime[neuron] - time) - 1
                    deltaT = deltaT / 100.0
                    deltaW = -1 * 2.71828**(-deltaT / .02)
                    weights[neuron][history[neuron][i]] += deltaW * learningRate
            historyCount[neuron] = 0
        cuda.syncthreads()

    #this is what the user calls
    def advanceTime(self, layerIndex, timeIndex, blocks, threads, thresholdParam):

        self.advanceTimeCUDA[blocks, threads](self.spikes[layerIndex - 1],
            self.layersPotential[layerIndex],
            self.spikes[layerIndex],
            self.weights[layerIndex],
            layerIndex,
            self.inputTracker[layerIndex],
            self.inputTimes[layerIndex],
            self.inputTrackerCount[layerIndex],
            timeIndex,
            thresholdParam,
            self.spikeTime[layerIndex])
        cuda.synchronize()

    #wrapper function for the weight adjustment kernel
    def adjustWeightsWrapper(self, layerIndex, timeIndex, blocks, threads, STDP):
        self.adjustWeights[blocks, threads](layerIndex, self.weights[layerIndex],
            self.spikes[layerIndex],
            self.inputTracker[layerIndex],
            self.inputTimes[layerIndex],
            self.inputTrackerCount[layerIndex],
            self.learningRate,
            self.learningRate2,
            STDP,
            timeIndex,
            self.spikeTime[layerIndex])

        cuda.synchronize()

    #Weight initialization
    def createRandomWeights(self):

        for layer in self.weightsTemp:
            for row in layer:
                for i in range(0, len(row)):
                    #row[i] = np.random.normal(loc=.8, scale=.01)
                    row[i] = np.random.normal(loc=.0, scale=.8)
                    #row[i] = (2 * (random.random() - .5))

    #TODO this is not final yet
    def inputValues(self, values, STDP=True):
        #inputWindows = []
        #self.windowSize
        #print("before")
        #before = np.copy(self.weights[1].copy_to_host())
        #print(before)
        #print("after")
        #flags for kernels
        threadcount = 320
        blocks = 1
        #values = np.array(values)
        #print(self.weights[-1].copy_to_host())
        #loops through each time step
        values = cuda.to_device(values)

        for time in range(0, self.windowSize):
            #probably got this from creel tutorial
            blocks = (int)(self.spikes[0].copy_to_host().size / threadcount)
            if blocks * threadcount < self.spikes[0].copy_to_host().size:
                blocks += 1

            self.CUDAInput[blocks, threadcount](self.spikes[0], values, time)

            #propagates spikes through the network
            for layer in range(1, len(self.spikes)):
                blocks = (int)(self.spikes[layer].copy_to_host().size / threadcount)
                if blocks * threadcount < self.spikes[layer].copy_to_host().size:
                    blocks += 1
                self.advanceTime(layer, time, blocks, threadcount, self.threshold[layer - 1])

            #keeps track of the outputs on the output layer, and what times they occured
            blocks = (int)(self.spikes[-1].copy_to_host().size / threadcount)
            if blocks * threadcount < self.spikes[-1].copy_to_host().size:
                blocks += 1

            self.CUDAoutputTracker[blocks, threadcount](self.spikes[-1], self.outputTimes, time)

            #adjusts the weights, but only if STDP is enabled
            for layer in range(1, len(self.spikes)):
                blocks = (int)(self.spikes[layer].copy_to_host().size / threadcount)
                if blocks * threadcount < self.spikes[layer].copy_to_host().size:
                    blocks += 1
                self.adjustWeightsWrapper(layer, time, blocks, threadcount, STDP)

        #TODO after the for loop is done, reset all of the values
        #after = np.copy(self.weights[1].copy_to_host())

        #for i in after:
            #for j in i:
                #print(j)
        return self.outputTimes.copy_to_host()
        #we then need to loop through all of the neurons in the first layer
        #and adjust the correct values
