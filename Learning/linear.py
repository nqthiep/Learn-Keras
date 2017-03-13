from keras.layers import Dense
from keras.models import Sequential

from AbstractAI import AbstractAI
import numpy as np

class Linear(AbstractAI):
    def fx(self, x):
        return 2*x + 5

    def __prepareData(self, minValue, maxValue, count):
        x_tmp = []
        y_tmp = []
        for i in range(count):
            x = np.random.randint(minValue, maxValue)
            y = self.fx(x)
            x_tmp.append(x)
            y_tmp.append([x, y])
        return x_tmp, y_tmp

    def createData(self):
        self.x_data, self.y_data = self.__prepareData(0, 1000, 1000)
        self.x_test, self.y_test = self.__prepareData(0, 1000, 100)
        return self.x_data, self.y_data, self.x_test, self.y_test

    def createModel(self):
        model = Sequential()
        model.add(Dense(output_dim=2, input_dim=1))
        return model

linear = Linear()
linear.modelFile = './models/linear.hf5'
linear.loss = 'mse'
linear.optimizer = 'Adam'
linear.epoch = 1000

linear.start(retrain = True)
linear.predict(np.array([-100]))

linear.showData()
linear.showHistoryAccuracy()
linear.showHistoryLoss()