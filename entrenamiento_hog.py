import numpy as np

from neuronalnetwork.NeuralNetwork import NeuralNetwork
from neuronalnetwork.NeuralNetworkWith2HiddenLayers import NeuralNetworkWith2HiddenLayers
import h5py
import numpy
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog

from neuronalnetwork.NeuronalNetworkWith3HiddenLayers import NeuralNetworkWith3HiddenLayers

r1 = NeuralNetworkWith2HiddenLayers()
r1.layer_1 = 36
r1.layer_2 = 32
r1.layer_3 = 32
r1.layer_4 = 12
data = h5py.File("digitos.h5", "r")
r2 = NeuralNetwork()
r2.capa1 = 36
r2.capa2 = 32
r2.capa3 = 12
r3 = NeuralNetworkWith3HiddenLayers()
r3.layer_1 = 36
r3.layer_2 = 16
r3.layer_3 = 16
r3.layer_4 = 16
r3.layer_5 = 12

lista_hog = []
for x in data["X"][:]:
    descriptor = hog(x.reshape(28, 28), orientations=9, pixels_per_cell=(10,10), cells_per_block=(1, 1))
    lista_hog.append(descriptor)


def get_descriptor(file_path, flag):
    imagen = cv2.imread(file_path)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_gris = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
    ret, imagen_bn = cv2.threshold(imagen_gris, 90, 255, cv2.THRESH_BINARY_INV)
    grupos, _ = cv2.findContours(imagen_bn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ventanas = [cv2.boundingRect(g) for g in grupos]
    hog_list = []
    for v in ventanas:
        cv2.rectangle(imagen, (v[0], v[1]), (v[0] + v[2], v[1] + v[3]), (255, 0, 0), 2)
        espacio = int(v[3] * 1.6)
        p1 = int((v[1] + v[3] // 2) - espacio // 2)
        p2 = int((v[0] + v[2] // 2) - espacio // 2)
        digito = imagen_bn[p1:p1 + espacio, p2:p2 + espacio]
        if flag == True:
            if p2 > 0 and p1 > 0 and 50 < espacio < 400:
                digito = cv2.resize(digito, (28, 28), interpolation=cv2.INTER_AREA)
                descriptor = hog(digito, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(1, 1))
                aux = descriptor.reshape(1, -1)
                hog_list.append(aux)
        else:
            if p2 > 0 and p1 > 0:
                digito = cv2.resize(digito, (28, 28), interpolation=cv2.INTER_AREA)
                descriptor = hog(digito, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(1, 1))
                aux = descriptor.reshape(1, -1)
                hog_list.append(aux)
    return hog_list


descriptores = numpy.array(lista_hog)
plus_descriptors1 = numpy.array(get_descriptor("plusdataset/ggg.jpeg", True))
plus_descriptors_aux1 = plus_descriptors1.reshape((14, 36))
plus_descriptors2 = numpy.array(get_descriptor("plusdataset/gg.jpg", True))
plus_descriptors_aux2 = plus_descriptors2.reshape((plus_descriptors2.shape[0], plus_descriptors2.shape[2]))
substraction_descriptors1 = numpy.array(get_descriptor("substractionset/substraction_dataset_1.jpeg", False))
substraction_descriptors_aux1 = substraction_descriptors1.reshape((substraction_descriptors1.shape[0], substraction_descriptors1.shape[2]))
substraction_descriptors2 = numpy.array(get_descriptor("substractionset/substraction_dataset_2.jpeg", False))
substraction_descriptors_aux2 = substraction_descriptors2.reshape((substraction_descriptors2.shape[0], substraction_descriptors2.shape[2]))
r1.fit(
    np.concatenate([substraction_descriptors_aux1,
                    plus_descriptors_aux2,
                    descriptores,
                    plus_descriptors_aux1,
                    substraction_descriptors_aux2]),
    np.concatenate([
        (np.ones((len(substraction_descriptors1)), dtype=np.int32) * 11),
        (np.ones((len(plus_descriptors2)), dtype=np.int32) * 10),
        data["y"][:],
        (np.ones((len(plus_descriptors1)), dtype=np.int32) * 10),
        (np.ones((len(substraction_descriptors_aux2)), dtype=np.int32) * 11),
    ], dtype=np.int32))
r2.fit(
    np.concatenate([substraction_descriptors_aux1,
                    plus_descriptors_aux2,
                    descriptores,
                    plus_descriptors_aux1,
                    substraction_descriptors_aux2]),
    np.concatenate([
        (np.ones((len(substraction_descriptors1)), dtype=np.int32) * 11),
        (np.ones((len(plus_descriptors2)), dtype=np.int32) * 10),
        data["y"][:],
        (np.ones((len(plus_descriptors1)), dtype=np.int32) * 10),
        (np.ones((len(substraction_descriptors_aux2)), dtype=np.int32) * 11),
    ], dtype=np.int32))
r3.fit(
    np.concatenate([substraction_descriptors_aux1,
                    plus_descriptors_aux2,
                    descriptores,
                    plus_descriptors_aux1,
                    substraction_descriptors_aux2]),
    np.concatenate([
        (np.ones((len(substraction_descriptors1)), dtype=np.int32) * 11),
        (np.ones((len(plus_descriptors2)), dtype=np.int32) * 10),
        data["y"][:],
        (np.ones((len(plus_descriptors1)), dtype=np.int32) * 10),
        (np.ones((len(substraction_descriptors_aux2)), dtype=np.int32) * 11),
    ], dtype=np.int32))
r1.initialize_thetas()
r2.initialize_thetas()
r3.initialize_thetas()
r1.train("thetas_hog_entrenamiento1.h5")
r2.train("thetas_hog_entrenamiento2.h5")
r3.train("thetas_hog_entrenamiento3.h5")
