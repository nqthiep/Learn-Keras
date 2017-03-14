from keras.layers import Dense
from keras.models import Sequential

from AbstractAI import AbstractAI
import numpy as np

class Linear(AbstractAI):
    def fx1(self, x1, x2, x3):
        return x1 + x2 + x3

    def fx2(self, x1, x2, x3):
        return x1 - x2 - x3

    def __prepareData(self, minValue, maxValue, count):
        x_tmp = []
        y_tmp = []
        for i in range(count):
            x1 = np.random.randint(minValue, maxValue)
            x2 = np.random.randint(minValue, maxValue)
            x3 = np.random.randint(minValue, maxValue)
            y1 = self.fx1(x1, x2, x3)
            y2 = self.fx2(x1, x2, x3)
            x_tmp.append([x1, x2, x3])
            y_tmp.append([y1, y2])
        return x_tmp, y_tmp

    def createData(self):
        self.x_data, self.y_data = self.__prepareData(0, 1000, 1000)
        self.x_test, self.y_test = self.__prepareData(0, 1000, 100)
        return self.x_data, self.y_data, self.x_test, self.y_test

    def createModel(self):
        model = Sequential()
        model.add(Dense(output_dim=2, input_dim=3))
        return model

linear = Linear()
linear.modelFile = './models/linear.hf5'
linear.loss = 'mse'
linear.optimizer = 'Adam'
linear.epoch = 10000

linear.start(retrain = True)
linear.predict(np.array([[100, 200, 300]]))

# linear.showData()
# linear.showHistoryAccuracy()
# linear.showHistoryLoss()