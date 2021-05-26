import cv2
from skimage.feature import hog

from calculator import Calculator
from neuronalnetwork.NeuralNetwork import NeuralNetwork
from neuronalnetwork.NeuralNetworkWith2HiddenLayers import NeuralNetworkWith2HiddenLayers

r = NeuralNetworkWith2HiddenLayers()
r.load_data("thetas_hog_entrenamiento1.h5")
imagen = cv2.imread("casos/aa.jpeg")
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagen_gris = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
ret, imagen_bn = cv2.threshold(imagen_gris, 90, 255, cv2.THRESH_BINARY_INV)
grupos, _ = cv2.findContours(imagen_bn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ventanas = [cv2.boundingRect(g) for g in grupos]
to_process = {}
calculator = Calculator()
for v in ventanas:
    cv2.rectangle(imagen, (v[0], v[1]), (v[0] + v[2], v[1] + v[3]), (255, 0, 0), 2)
    espacio = int(v[3] * 1.6)
    p1 = int((v[1] + v[3] // 2) - espacio // 2)
    p2 = int((v[0] + v[2] // 2) - espacio // 2)
    digito = imagen_bn[p1:p1 + espacio, p2:p2 + espacio]
    if p2 > 0 and p1 > 0 and v[2] > 10 and v[3] > 10 and v[2] < 200 and v[3] < 200:
        digito = cv2.resize(digito, (28, 28), interpolation=cv2.INTER_AREA)
        descriptor = hog(digito, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(1,1))
        aux = descriptor.reshape(1, -1)
        prediccion = r.predict(aux)
        cv2.putText(imagen, str(prediccion[0]), (v[0], v[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
        to_process[v[0]] = prediccion[0]
print("El resultado es ", calculator.calculate(to_process))
cv2.imshow("Digitos", imagen)
cv2.waitKey(),

