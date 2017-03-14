from abc import ABC, abstractmethod
from keras.models import load_model
import matplotlib.pyplot as plt

class AbstractAI(ABC):
    def __init__(self):
        self.modelFile = None
        self.model = None
        self.x_data = None
        self.y_data = None
        self.x_test = None
        self.y_test = None
        self.epoch = 1000
        self.loss = None
        self.optimizer = None
        self.metrics = ['accuracy']
        self.history = None

    def start(self, retrain=False):
        if (retrain) or (not self.load()):
            self.x_data, self.y_data, self.x_test, self.y_test = self.createData()
            print(self.x_data, self.y_data, self.x_test, self.y_test)
            self.model = self.createModel()
            self.train()
            self.save()

    def load(self):
        try:
            self.model = load_model(self.modelFile)
            return True
        except Exception as e:
            print('Load model has an exeption: ', e)
            pass
        return False

    def save(self):
        if self.model is not None and self.modelFile is not None:
            try:
                self.model.save(self.modelFile)
            except Exception as e:
                print('Save model has an exception: ', e)

    def train(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model.summary()
        self.history = self.model.fit(self.x_data, self.y_data, nb_epoch=self.epoch, validation_data=(self.x_test, self.y_test))

    def predict(self, data):
        result = self.model.predict(data)
        print('predict({})={}', data, result)
        return result

    def showData(self):
        if self.x_data is not None and self.y_data is not None:
            plt.plot(self.x_data, self.y_data, 'r--')
            plt.show()

    def showHistoryAccuracy(self):
        if self.history:
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

    def showHistoryLoss(self):
        if self.history:
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

    @abstractmethod
    def createData(self):
        pass

    @abstractmethod
    def createModel(self):
        pass
