from scipy.optimize import minimize
import h5py
import numpy as np


class NeuralNetworkWith2HiddenLayers:
    def __init__(self):
        self.x = None
        self.y = None
        self.theta_1 = None
        self.theta_2 = None
        self.theta_3 = None
        self.lambda_ = 1
        self.epsilon: int = 0
        self.layer_1: int = 0  # input layer
        self.layer_2: int = 0  # hidden layer
        self.layer_3: int = 0  # output layer
        self.layer_4: int = 0

    def fit(self, x, y):
        self.x = x
        self.y = y

    def initialize_thetas(self):
        self.epsilon = 0.16
        self.theta_1 = np.random.rand(self.layer_2, (self.layer_1 + 1)) * 2 * self.epsilon - self.epsilon
        self.theta_2 = np.random.rand(self.layer_3, (self.layer_2 + 1)) * 2 * self.epsilon - self.epsilon
        self.theta_3 = np.random.rand(self.layer_4, (self.layer_3 + 1)) * 2 * self.epsilon - self.epsilon

    def sigmoid(self, number):
        return 1 / (1 + np.exp(-number))

    def sigmoid_derivative(self, number):
        return self.sigmoid(number) * (1 - self.sigmoid(number))

    def cost_function(self, t):
        t1 = np.reshape(t[0: self.layer_2 * (self.layer_1 + 1)], (self.layer_2, (self.layer_1 + 1)))
        t2 = np.reshape(t[self.layer_2 * (self.layer_1 + 1): self.layer_2 * (self.layer_1 + 1) + self.layer_3 * (self.layer_2 + 1)], (self.layer_3, (self.layer_2 + 1)))
        t3 = np.reshape(t[self.layer_2 * (self.layer_1 + 1) + self.layer_3 * (self.layer_2 + 1):], (self.layer_4, (self.layer_3 + 1)))
        m, n = self.x.shape
        a1 = np.concatenate([np.ones((m, 1)), self.x], axis=1)
        a2 = self.sigmoid(a1.dot(t1.T))
        a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
        a3 = self.sigmoid(a2.dot(t2.T))
        a3 = np.concatenate([np.ones((a3.shape[0], 1)), a3], axis=1)
        h = self.sigmoid(a3.dot(t3.T))
        y_vec = np.eye(self.layer_4)[self.y.reshape(-1)]
        param_reg = (self.lambda_ / (2 * m)) * (np.sum(np.square(t1[:, 1:])) +
                                                np.sum(np.square(t2[:, 1:])) +
                                                np.sum(np.square(t3[:, 1:])))

        j = - 1 / m * np.sum(np.log(h) * y_vec + np.log(1 - h) * (1 - y_vec)) + param_reg
        delta4 = h - y_vec
        delta3 = delta4.dot(t3)[:, 1:] * self.sigmoid_derivative(a2.dot(t2.T))
        delta2 = delta3.dot(t2)[:, 1:] * self.sigmoid_derivative(a1.dot(t1.T))
        delta_acum_1 = delta2.T.dot(a1)
        delta_acum_2 = delta3.T.dot(a2)
        delta_acum_3 = delta4.T.dot(a3)
        grad1 = 1 / m * delta_acum_1
        grad2 = 1 / m * delta_acum_2
        grad3 = 1 / m * delta_acum_3
        grad1[:, 1:] = grad1[:, 1:] + (self.lambda_ / m) * t1[:, 1:]
        grad2[:, 1:] = grad2[:, 1:] + (self.lambda_ / m) * t2[:, 1:]
        grad3[:, 1:] = grad3[:, 1:] + (self.lambda_ / m) * t3[:, 1:]
        grad = np.concatenate([grad1.flatten(), grad2.flatten(), grad3.flatten()])
        return j, grad

    def train(self, destiny):
        j_grad = lambda p: self.cost_function(p)
        theta_inical = np.concatenate([self.theta_1.flatten(), self.theta_2.flatten(), self.theta_3.flatten()])
        opciones = {'maxiter': 2000}
        res = minimize(j_grad, theta_inical, jac=True, method="TNC", options=opciones)
        theta_optimo = res.x
        self.theta_1 = np.reshape(theta_optimo[0:self.layer_2 * (self.layer_1 + 1)], (self.layer_2, (self.layer_1 + 1)))
        self.theta_2 = np.reshape(theta_optimo[self.layer_2 * (self.layer_1 + 1): self.layer_2 * (self.layer_1 + 1) + self.layer_3 * (self.layer_2 + 1)], (self.layer_3, (self.layer_2 + 1)))
        self.theta_3 = np.reshape(theta_optimo[(self.layer_2 * (self.layer_1 + 1)) + (self.layer_3 * (self.layer_2 + 1)):], (self.layer_4, (self.layer_3 + 1)))
        arch = h5py.File(destiny, "w")
        arch.create_dataset("Theta1", data=self.theta_1)
        arch.create_dataset("Theta2", data=self.theta_2)
        arch.create_dataset("Theta3", data=self.theta_3)

    def predict(self, image):
        a1 = np.concatenate([np.ones((1, 1)), image], axis=1)
        a2 = self.sigmoid(a1.dot(self.theta_1.T))
        a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
        a3 = self.sigmoid(a2.dot(self.theta_2.T))
        a3 = np.concatenate([np.ones((a3.shape[0], 1)), a3], axis=1)
        a4 = self.sigmoid(a3.dot(self.theta_3.T)).T
        return a4.argmax(), a4[a4.argmax()]

    def load_data(self, file):
        loaded_file = h5py.File(file, "r")
        self.theta_1 = loaded_file["Theta1"][:]
        self.theta_2 = loaded_file["Theta2"][:]
        self.theta_3 = loaded_file["Theta3"][:]
