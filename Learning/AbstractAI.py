from abc import ABC, abstractmethod
from keras.models import load_model


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

    def start(self, retrain=False):
        if (retrain) or (not self.load()):
            self.x_data, self.y_data, self.x_test, self.y_test = self.createData()
            self.model = self.createModel()
            self.train()
            self.save()

    def load(self):
        try:
            self.model = load_model(self.modelFile)
            return True
        except:
            print('Load model has an exeption!')
            pass
        return False

    def save(self):
        if self.model is not None and self.modelFile is not None:
            try:
                self.model.save(self.modelFile)
            except:
                print('Save model has an exception!')

    def train(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model.fit(self.x_data, self.y_data, nb_epoch=self.epoch, validation_data=(self.x_test, self.y_test))

    def predict(self, data):
        return self.model.predict(data)

    @abstractmethod
    def createData(self):
        pass

    @abstractmethod
    def createModel(self):
        pass
