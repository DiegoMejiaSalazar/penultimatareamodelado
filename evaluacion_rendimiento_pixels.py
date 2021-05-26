from neuronalnetwork.NeuralNetwork import NeuralNetwork
from neuronalnetwork.NeuralNetworkWith2HiddenLayers import NeuralNetworkWith2HiddenLayers
from skimage.feature import hog
import h5py
import matplotlib.pyplot as plt

from neuronalnetwork.NeuronalNetworkWith3HiddenLayers import NeuralNetworkWith3HiddenLayers

r = NeuralNetworkWith2HiddenLayers()
r.load_data("thetas_hog_entrenamiento1.h5")
testing_data = h5py.File("digitos_test.h5", "r")
r2 = NeuralNetwork()
r2.load_data("thetas_hog_entrenamiento2.h5")
r3 = NeuralNetworkWith3HiddenLayers()
r3.load_data("thetas_hog_entrenamiento3.h5")
lista_hog = []
corrected1=0
corrected2=0
corrected3=0
for x,y in zip(testing_data["X"][:], testing_data["y"][:]):
    predicted_result, _ = r.predict(x.reshape(1, -1))
    predicted_result2, _ = r2.predict(x.reshape(1, -1))
    predicted_result3, _ = r3.predict(x.reshape(1, -1))
    if predicted_result == y:
        corrected1 += 1
    if predicted_result2 == y:
        corrected2 += 1
    if predicted_result3 == y:
        corrected3 += 1
print("Corrects (3 hidden layers): ", (corrected3 * 100) / len(testing_data["y"][:]), "%")
print("Corrects (2 hidden layers): ", (corrected1 * 100) / len(testing_data["y"][:]), "%")
print("Corrects (1 hidden layer): ", (corrected2 * 100) / len(testing_data["y"][:]), "%")
