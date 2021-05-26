# Trabajo modelado
Estudiante: Diego Orlando Mejia Salazar
## Resumen
En el presente trabajo se hizo la implementación de dos redes neuronales las cuales difieren en el numero de capas en el hidden layer a ambas redes neuronales se les hizo un par de modificaciones la primera fue en la capa de entrada haciendo que cada red neuronal reciba los pixeles de cada imagen en el caso de la imagen de 28*28 pixeles (784 pixeles en total) luego se vio que este numero iba a ser demasiado grande al usar una imagen de mas calidad por lo que se hizo la prueba usando el descriptor HOG lo cual hizo que nuestra capa de entrada recibiera 36 entradas lo cual es mucho mas aceptable que 784, posteriormente se paso a entrenar la red modificando el numero de neuronas en las hidden layers para posteriormente evaluar el rendimiento, también se procedió a crear un dataset de entrenamiento para que la red neuronal sea capaz de reconocer los signos + y – lo cual implico modificaciones en el output-layer

## Introducción
Los modelos de aprendizaje automática son una técnica matemática para representar ciertos patrones que se encontraban en los datos (en este caso los dígitos y los signos + y -), ya que gracias a estos la red neuronal es capaz de descubrir ciertas estructuras en los datos que resultan ser decisivas para hacer una predicción, en este caso se utilizó la regresión logística (función sigmoidal Y = 1 / 1+e -z) ya que se enfrento a un problema de clasificación entre los dígitos del 0 al 9 y los signos + y -  

## Fundamentación
Las redes neuronales no son mas que un conjunto de neuronas las cuales principalmente se dividen en capaz (input layer, hidden layer y output layer) estas capas se componen de neuronas y dichas capas se comunican entre si a través de sus pesos(weights) y biases(sesgo), esto con el fin de proporcionar una predicción dado un conjunto de características
-   ¿Cómo funciona?
La mejor forma de describir el funcionamiento de la red neuronal es a base de dos algorimos forward propagation y backward propagation
### Forward propagation
- Este algoritmo se caracteriza que para una entrada X la cual es un conjunto de características de un dato, esta se propaga hacia adelante en este caso a los hidden layers y por último se propaga hacia el output layer, esta información es escalada o multiplicada por los weights o pesos de una neurona h hacia otra neurona i, y esta información se le adicionara o sumara un valor (bias o sesgo)esto se puede representar con la siguiente imagen
![image](https://user-images.githubusercontent.com/80376417/119605562-024dfa00-bdbf-11eb-8cfc-429d2982c2b6.png)
Esta sumatoria se le pasa a una función de activación la cual en este caso es la función sigmoidal que devuelve valores entre 1 y 0, esto es importante ya que en el caso de que la funcion sigmoidal devuelva 1 significa que los datos proporcionados a esta son relevantes para tomar una predicción pero en el caso de que sea 0 significa que el dato es irelevante para hacer una predicción
### Backward propagation
El algoritmo backward propagation se utiliza principalmente debido a que la red neuronal se inicializa con weights y biases randomicos lo cual tienen que ser posteriormente ajustados para hacer una predicción más certera, este algoritmo a diferencial del forward propagation empieza del output layer y termina en la primera capa del hidden layer, este algoritmo utiliza en este caso la entropía cruzada que nos sirve para estima el error y hacer un reajuste de los pesos y biases, esta entropía cruzada utiliza a su vez la derivada de la función sigmoidal para el cálculo del error entre capas, el cálculo de la entropía cruzada es el siguiente:

![image](https://user-images.githubusercontent.com/80376417/119605679-31fd0200-bdbf-11eb-99fb-f6d44bf46308.png)

La cantidad de vez que se hace este backward propagation se definió en 2000 veces, cuanto mas veces se hace(epoch) el error va a ser menor pero va a tardar más en realizar el entrenamiento

![image](https://user-images.githubusercontent.com/80376417/119605701-3cb79700-bdbf-11eb-97c5-90c338868c09.png)

## Experimentos
En los primeros incisos hace referencia al entrenamiento (entrenamiento_pixels.py, entrenamiento_hog.py) y la evaluación (evaluacion_rendimiento_pixels.py, evaluación_rendimiento_hog.py) que se hace a la hora de realizar una predicción esto usando tanto pixeles como el descriptor HOG, por lo tanto se mostrara los siguientes resultados

### Descriptor HOG
36 neuronas en input layer, 16 neuronas en el hidden layer (1 capa), 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605757-535dee00-bdbf-11eb-9126-9f837ea6fa2f.png)

36 neuronas en input layer, 16 neuronas en el hidden layer (primera capa del hidden layer), 16 neuronas en el hidden layer(segunda capa del hidden layer), 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605782-5bb62900-bdbf-11eb-8b31-c7656847c8de.png)

36 neuronas en input layer, 16 neuronas en el hidden layer (primera capa del hidden layer), 16 neuronas en el hidden layer (segunda capa del hidden layer), 16 neuronas en el hidden layer(tercera capa del hidden layer , 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605810-6a044500-bdbf-11eb-8b08-215a127e4d30.png)

36 neuronas en input layer, 32 neuronas en el hidden layer (1 capa), 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605836-72f51680-bdbf-11eb-9097-9ea2444aa439.png)

36 neuronas en input layer, 32 neuronas en el hidden layer (primera capa del hidden layer), 32 neuronas en el hidden layer(segunda capa del hidden layer), 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605853-7b4d5180-bdbf-11eb-97be-df3de4ddae2a.png)


36 neuronas en input layer, 32 neuronas en el hidden layer (primera capa del hidden layer), 32 neuronas en el hidden layer(segunda capa del hidden layer), 32 neuronas en el hidden layer(tercera capa del hidden layer),  12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605875-84d6b980-bdbf-11eb-825d-2d701781f908.png)


## Usando solamente pixeles
784 neuronas en input layer, 32 neuronas en el hidden layer (1 capa), 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605899-915b1200-bdbf-11eb-860c-c50daf513a9a.png)

784 neuronas en input layer, 32 neuronas en el hidden layer (primera capa del hidden layer), 32 neuronas en el hidden layer (segunda capa del hidden layer), 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605914-99b34d00-bdbf-11eb-8ada-6f44ce24c10e.png)

784 neuronas en input layer, 32 neuronas en el hidden layer (primera capa del hidden layer), 32 neuronas en el hidden layer (segunda capa del hidden layer), 32 neuronas en el hidden layer (tercera capa del hidden layer), 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605933-a6d03c00-bdbf-11eb-88e0-b9f34cfc3d5c.png)

784 neuronas en input layer, 16 neuronas en el hidden layer (1 capa), 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605961-b059a400-bdbf-11eb-820b-48c06151a7e9.png)


784 neuronas en input layer, 16 neuronas en el hidden layer (primera capa del hidden layer), 16 neuronas en el hidden layer(segunda capa del hidden layer), 12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119605976-b8194880-bdbf-11eb-97f1-1a18bfa3e734.png)


784 neuronas en input layer, 16 neuronas en el hidden layer (primera capa del hidden layer), 16 neuronas en el hidden layer(segunda capa del hidden layer), 16 neuronas en el hidden layer(tercera capa del hidden layer),  12 neuronas en el output layer

![image](https://user-images.githubusercontent.com/80376417/119606022-cc5d4580-bdbf-11eb-8a5c-6dc13e0fe4b9.png)

Posteriormente se procedió a hacer predicciones de fotografías de números con signos logrando que pueda reconocer tanto números como signos, el signo + se le etiqueto con el valor de 10 y el signo – se etiqueto con el valor de 11 y se obtuvieron los siguientes resultados 

![image](https://user-images.githubusercontent.com/80376417/119606069-dda65200-bdbf-11eb-9860-20aceec96b79.png)

El programa también es capaz de realizar esta sumatoria dando el siguiente resultado, este algoritmo se implemento en el archivo Calculator.py:

![image](https://user-images.githubusercontent.com/80376417/119606084-e4cd6000-bdbf-11eb-9708-990cf1dd1a77.png)

También se hizo otra prueba con la siguiente imagen :


![image](https://user-images.githubusercontent.com/80376417/119606126-f6af0300-bdbf-11eb-9003-ce5fad16cfe6.png)

Obteniendo el resultado de 

![image](https://user-images.githubusercontent.com/80376417/119606146-02022e80-bdc0-11eb-83aa-f155c67daa3b.png)

Por ultimo se hizo un filtro adicional para que solo se reconozcan dígitos y signos que no sean demasiado grandes ni tampoco demasiado pequeños esto se hizo mediante el uso de una condicional que evalúa el ancho y el alto de la ventana que enmarca el digito, de esta forma solo se pasan a procesar los dígitos que tengan un máximo de 200 de alto y 200 de ancho y un mínimo de 10 de alto y 10 de ancho:

![image](https://user-images.githubusercontent.com/80376417/119606159-0a5a6980-bdc0-11eb-8795-c26e88873f04.png)

#### Nota: Los datasets para hacer el reconocimiento de los signos de adicion y substraccion se encuentra en la carpeta substractionset y plusdataset

# Conclusion

Se vio que cuando la red neuronal es entrenada recibiendo pixeles en el input layer tarda entre 2 a 3 horas en cambio cuando la red neuronal recibe el H.O.G. descriptor en el input layer la red neuronal tarda en ser entrenada un estimado de 10 a 15 minutos, esto es debido a que cuando la red neuronal recibe pixeles en la capa de entrada las operaciones matriciales se realizan con matrices de mayores dimensiones en el caso de recibir los pixeles de una imagen de 28 * 28 pixeles se trabaja con matrices cuyas filas tienen 784 elementos más 1 elemento extra el cual es el bias, en cambio al usar el H.O.G. descriptor esta cantidad de elementos se reduce a 36 más 1 elemento extra el cual es el bias,  también se vio que a mayor cantidad de iteraciones de backward propagation la red neuronal es mucho más precisa pero esto va de la mano con el parámetro de regularización y la topología ya que la red neuronal con tres capas ocultas presento el peor resultado frente a una red neuronal de 2 o 1 capa oculta y esto es debido a que si no se ajusta adecuadamente los parámetros de regularización esta puede presentar un fenómeno underfitting la cual hace que nuestro modelo no se puede adecuar apropiadamente a los datos

![image](https://user-images.githubusercontent.com/80376417/119606192-1cd4a300-bdc0-11eb-9a5f-079a1795db6b.png)

En la siguiente grafica se describe en el eje x las topologías de las redes neuronales en el siguiente formato
<HOG DESCRIPTOR/PIXEL> <X (Cantidad de neuronas en el input layer)> - <B(Cantidad de neuronas en la primera capa del hidden layer)> - <C (Cantidad de neuronas en la segunda capa del hidden layer)> … <Y Cantidad de neuronas en el output layer>, por ultimo me gustaría calificar a la practica con un puntaje de 5/5 debido a que me fue bastante útil y aprendí demasiado. Gracias





