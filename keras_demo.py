import keras
from keras import layers, models
from tensorflow.examples.tutorials.mnist import input_data

class ConvNet():
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()
        self._build_optimizer()

    def build_model(self):
        input_image = layers.Input(self.config['image_size'])
        conv1 = layers.Conv2D(filters=12, kernel_size=3, strides=1)(input_image)
        bn1 = layers.BatchNormalization(axis=3)(conv1)
        relu1 = layers.Activation('relu')(bn1)
        pool1 = layers.MaxPool2D(pool_size=2, strides=1)(relu1)

        conv2 = layers.Conv2D(filters=24, kernel_size=3, strides=1)(pool1)
        bn2 = layers.BatchNormalization(axis=3)(conv2)
        relu2 = layers.Activation('relu')(bn2)
        pool2 = layers.MaxPool2D(pool_size=2, strides=1)(relu2)

        flatten = layers.Flatten()(pool2)
        dense1 = layers.Dense(units=256, activation='relu')(flatten)
        output = layers.Dense(units=self.config["num_classes"], activation='softmax')(dense1)

        model = models.Model(input_image, output)
        return model

    def _build_optimizer(self):
        self.loss = keras.losses.categorical_crossentropy
        self.metrics = [keras.metrics.categorical_accuracy]
        self.optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(loss=self.loss, optimizer=self.optimizer,metrics=['accuracy'])

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y, batch_size=self.config['batch_size'], epochs=self.config['epochs'], verbose=1)

    def evaluate(self, test_x, test_y):
        score = self.model.evaluate(test_x, test_y, verbose=0)
        print("loss:{}; accuracy:{}".format(score[0], score[1]))


def get_config():
    config = {"learning_rate":0.001,
              "epochs":5,
              "batch_size":64,
              "num_classes":10,
              "image_size":(28,28,1)}
    return config

if __name__ == "__main__":
    config = get_config()
    model = ConvNet(config)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.labels
    train_x = train_x.reshape((-1, 28, 28, 1))
    test_x = test_x.reshape((-1, 28, 28, 1))
    model.train(train_x, train_y)
    model.evaluate(test_x, test_y)
