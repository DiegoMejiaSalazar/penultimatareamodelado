import h5py
import numpy
from scipy.optimize import minimize


class NeuralNetwork:

    def __init__(self):
        self.X = None
        self.y = None
        self.theta1 = None
        self.theta2 = None
        self.lambda_ = 1
        self.capa1 = None
        self.capa2 = None
        self.capa3 = None

    def fit(self, x, y):
        self.X = x
        self.y = y

    def initialize_thetas(self, epsilon=0.12):
        self.theta1 = numpy.random.rand(self.capa2, (self.capa1 + 1)) * 2 * epsilon - epsilon
        self.theta2 = numpy.random.rand(self.capa3, (self.capa2 + 1)) * 2 * epsilon - epsilon

    @staticmethod
    def sigmoide(z):
        return 1 / (1 + numpy.exp(-z))

    def derivada_sigmoide(self, z):
        return self.sigmoide(z) * (1 - self.sigmoide(z))

    def funcion_costo_gradiente(self, t):
        t1 = numpy.reshape(t[0:self.capa2 * (self.capa1 + 1)], (self.capa2, (self.capa1 + 1)))
        t2 = numpy.reshape(t[self.capa2 * (self.capa1 + 1):], (self.capa3, (self.capa2 + 1)))

        m, n = self.X.shape

        a1 = numpy.concatenate([numpy.ones((m, 1)), self.X], axis=1)
        a2 = self.sigmoide(a1.dot(t1.T))
        a2 = numpy.concatenate([numpy.ones((a2.shape[0], 1)), a2], axis=1)
        h = self.sigmoide(a2.dot(t2.T))

        y_vec = numpy.eye(self.capa3)[self.y.reshape(-1)]

        param_reg = (self.lambda_ / (2 * m)) * (numpy.sum(numpy.square(t1[:, 1:])) +
                                                numpy.sum(numpy.square(t2[:, 1:])))

        j = - 1 / m * numpy.sum(numpy.log(h) * y_vec + numpy.log(1 - h) * (1 - y_vec)) + param_reg
        delta3 = h - y_vec
        delta2 = delta3.dot(t2)[:, 1:] * self.derivada_sigmoide(a1.dot(t1.T))
        delta_acum_1 = delta2.T.dot(a1)
        delta_acum_2 = delta3.T.dot(a2)
        grad1 = 1 / m * delta_acum_1
        grad2 = 1 / m * delta_acum_2
        grad1[:, 1:] = grad1[:, 1:] + (self.lambda_ / m) * t1[:, 1:]
        grad2[:, 1:] = grad2[:, 1:] + (self.lambda_ / m) * t2[:, 1:]

        grad = numpy.concatenate([grad1.flatten(), grad2.flatten()])
        return j, grad

    def train(self, destino):
        j_grad = lambda p: self.funcion_costo_gradiente(p)
        theta_inical = numpy.concatenate([self.theta1.flatten(), self.theta2.flatten()])
        opciones = {'maxiter': 2000}
        res = minimize(j_grad, theta_inical, jac=True, method="TNC", options=opciones)
        theta_optimo = res.x
        self.theta1 = numpy.reshape(theta_optimo[0:self.capa2 * (self.capa1 + 1)], (self.capa2, (self.capa1 + 1)))
        self.theta2 = numpy.reshape(theta_optimo[self.capa2 * (self.capa1 + 1):], (self.capa3, (self.capa2 + 1)))
        arch = h5py.File(destino, "w")
        arch.create_dataset("Theta1", data=self.theta1)
        arch.create_dataset("Theta2", data=self.theta2)

    def predict(self, imagen):
        a1 = numpy.concatenate([numpy.ones((1, 1)), imagen], axis=1)
        a2 = self.sigmoide(a1.dot(self.theta1.T))
        a2 = numpy.concatenate([numpy.ones((a2.shape[0], 1)), a2], axis=1)
        a3 = self.sigmoide(a2.dot(self.theta2.T)).T
        return a3.argmax(), a3[a3.argmax()]

    def load_data(self, archivo):
        arch = h5py.File(archivo, "r")
        self.theta1 = arch["Theta1"][:]
        self.theta2 = arch["Theta2"][:]
