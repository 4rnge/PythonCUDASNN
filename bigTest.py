from SNN_CUDA import SpikingCudaNetwork
import random
import numpy as np
import tensorflow as tf
import datetime
from libsvm.svmutil import *


#test comment
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

outputLayerSize = 50

#layer sizes, size of input window, and learning rate
#KEEP THIS SO I DONT FORGET :P
SNN = SpikingCudaNetwork([784, 1000, 300, outputLayerSize], 100, 101, .003, 0, threshold=[7, 3, 2])
#SNN = SpikingCudaNetwork([outputLayerSize], 100, 101, .04, 0, threshold = 1)


#a dictionary to store the outputed values
#it is going to store the input key(an integer)
#and the value is going to be a set of all outputs as an array
# example: dict[1] = {[0], [1], [3]}
#above is all outputs from 1
results = dict()


#generate the batches of size 50
batches = []
for i in range(0, 16000, 50):
    batches.append(i)


print("started at " + str(datetime.datetime.now()))

#the number of epochs it will run the same section 10 times
for e in range(0, 1):
    #if 1 < 0:
    random.shuffle(batches)

    print("running epoch " + str(e))

    counter = 0
    #loops through each starting offset
    for j in range(0, 320):
        offset = batches[j]
        print(str(counter))
        counter += 1
        #loops through the images in the selected batch
        for i in range(0 + offset, 50 + offset):

            inputs = []
            for x in x_train[i]:
                for y in x:
                    if y <= .1:
                        inputs.append(-1)
                    else:
                        inputs.append((int(99 - int(y * 99.0))))

            #print(str(y_train[i]) + " test " + str(i) + " epoch " + str(e) )

            output = SNN.inputValues(inputs, STDP=True)
            SNN.reinitialize()
            minimum = 1000
            listOfSpikes = []

            #what is the minimum
            #for j in range(0, output.size):

#                #sets the new minimum value
#             #   if output[j] < minimum and output[j] > -1:
#              #      minimum = output[j]

            #for j in range(0, output.size):
#             #   if output[j] == minimum:
#              #      listOfSpikes.append(j)
            #print("list of spikes is " + str(listOfSpikes))

            #print(str(listOfSpikes) + " at time " + str(minimum))

    #TODO next I should make it so that it records a set, that holds how many times
    #each result happened
    #also this doesn't seem to record the values how I would like them to be recorded
    print("results")

    #this now prints for every epoch
    for key in results.keys():
        print(str(key) + " " + str(results[key]))

    #clears the dictionary
    results = dict()
print("ended at " + str(datetime.datetime.now()))

#choice = input("run tests? y/n")
if 1 > 0:  # or choice == "y":

    results = np.zeros((1000, outputLayerSize))
    labels = np.zeros(1000)
    trainingSize = 1000
    imageStart = 4000
    imageCount = 1000
    testSize = 1000
    for e in range(0, 1):
        print("running epoch " + str(e))
        for i in range(imageStart, imageStart + trainingSize):
            # for i in range(imageCount + 100, imageCount + testSize + 100):
            inputs = []
            for x in x_train[i]:
                for y in x:
                    if y <= .1:
                        inputs.append(-1)
                    else:
                        inputs.append((int(99 - int(y * 99.0))))

            #print(str(y_train[i]) + " test " + str(i) + " epoch " + str(e) )

            output = SNN.inputValues(np.array(inputs), STDP=False)
            SNN.reinitialize()
            minimum = 1000
            listOfSpikes = []

            #what is the minimum
            #for j in range(0, output.size):

#                #sets the new minimum value
#            #   if output[j] < minimum and output[j] > -1:
#             #      minimum = output[j]

            #for j in range(0, output.size):
#               #if output[j] == minimum:
#                   #listOfSpikes.append(j)

            #for spike in listOfSpikes:
#            #   results[i-imageStart][spike] = 1
#               #results[i-(imageCount+100)][spike] = 1
            for neuron in range(0, output.size):
                results[i - imageStart][neuron] = output[neuron]

            labels[i - imageStart] = y_train[i]
            #labels[i-(imageCount+100)] = y_train[i]
            #print(str(y_train[i]) + " " + str(listOfSpikes))

    m = svm_train(labels, results, '-c 1 -t 0')
    p_label, p_acc, p_val = svm_predict(labels, results, m)
    print(str(p_acc))

print("ended at " + str(datetime.datetime.now()))


name = input("input name for file")
#SNN.saveWeights(name)
