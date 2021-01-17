# this file is a little rough but it is for making layer creation a little easier
#https://www.w3schools.com/python/python_classes.asp
#https://www.w3schools.com/python/python_inheritance.asp

import numpy as np


class layer():
    pass


class ConvolutionalLayer(layer):
    def __init__(self, layerShape, kernelShape, mapCount, poolShape, poolKernelShape):
        self.layerShape = layerShape
        self.kernelShape = kernelShape
        self.mapCount = mapCount
        self.poolShape = poolShape
        self.poolKernelShape = poolKernelShape


class DeepLayer(layer):
    def __init__(self, shape):
        self.shape = shape


e = "e"
deep = DeepLayer("test")
cnn = ConvolutionalLayer("test2")

test = np.empty(shape=3, dtype=layer)
test[0] = deep
test[1] = cnn
for i in test:
    print(type(i))
