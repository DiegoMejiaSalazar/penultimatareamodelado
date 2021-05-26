from flask import Flask

app = Flask(__name__)

from neuronalnetwork.NeuralNetwork import NeuralNetwork
from neuronalnetwork.NeuralNetworkWith2HiddenLayers import NeuralNetworkWith2HiddenLayers

r = NeuralNetworkWith2HiddenLayers()
r.load_data("thetas_hog_entrenamiento1.h5")
@app.route('/')
def hello_world():
    return {
        "hello": "World"
    }


if __name__ == '__main__':
    app.run()
