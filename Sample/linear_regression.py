from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train = [1, 2, 3, 4]
y_train = [5, 6, 7, 8]

model = Sequential()

model.add(Dense(output_dim=1, input_dim=1))
model.add(Dense(output_dim=1, input_dim=1, bias=True, activation='linear'))
print('Model: ', model.input_shape, model.output_shape)

# Dense implements the operation:
# output = activation(dot(input, kernel) + bias)
#
# - activation is the element-wise activation function passed as the activation argument
#      softmax
#      elu
#      softplus
#      softsign
#      relu
#      tanh
#      sigmoid
#      hard_sigmoid
#      linear
# - kernel is a weights matrix created by the layer
# - bias is a bias vector created by the layer (only applicable if use_bias is True).

model.compile(loss='mse', optimizer='sgd')

# prints summary of the model to the terminal
model.summary()

model.fit(x_train, y_train, nb_epoch=1000)

y_predict = model.predict(np.array([10]))
print(y_predict)