import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np
from mxnet.gluon.data.vision import transforms

def load_data():
    def data_xform(data):
        """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
        return ndarray.moveaxis(data, 2, 0).astype('float32') / 255

    train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(data_xform)
    train_data = mx.gluon.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)
    test_data = mx.gluon.data.DataLoader(test_data, batch_size=32, shuffle=False)
    return train_data, test_data


class GluonSeq():
    def __init__(self):
        self.net = self._build_model()
        self.loss = self._build_loss()
        self.trainer = self._build_trainer(self.net)
    def _build_model(self):
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, strides=1, activation='relu'))
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=1))
            net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1, activation='relu'))
            net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            net.add(gluon.nn.Flatten())
            net.add(gluon.nn.Dense(128, activation="relu"))
            net.add(gluon.nn.Dense(64, activation="relu"))
            net.add(gluon.nn.Dense(10))
            #net.collect_params().initialize(mx.init.Normal(sigma=0.05), ctx=mx.cpu())
            net.initialize()
        return net

    def _build_loss(self):
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        return softmax_cross_entropy

    def _build_trainer(self, net):
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
        return trainer

    def train(self, train_data, eval_data):
        epochs = 10
        acc = mx.metric.Accuracy()
        for e in range(epochs):
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(mx.cpu())
                label = label.as_in_context(mx.cpu())
                with autograd.record():
                    output = self.net(data)
                    loss = self.loss(output, label)
                loss.backward()
                self.trainer.step(data.shape[0])
                curr_loss = ndarray.mean(loss).asscalar()
                prediction = ndarray.argmax(output, axis=1)
                acc.update(labels=label, preds=prediction)
                if i%200 == 0:
                    print("Epoch:{}, Step:{}, Loss:{:.4f}, Accuracy:{:.4f}"
                          .format(e, i, curr_loss, acc.get()[1]))

            eval_accu = self.evaluate(eval_data)
            print("epoch:{}, evaluate accuracy:{:.2f}".format(e, eval_accu))

    def evaluate(self, eval_data):
        acc = mx.metric.Accuracy()
        for eval_x, eval_y in eval_data:
            output = self.net(eval_x)
            prediction = ndarray.argmax(output, axis=1)
            acc.update(labels=eval_y, preds=prediction)
        return acc.get()[1]










class GluonBlock(gluon.nn.Block):
    def __init__(self, **kwargs):
        super(GluonBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels=12, kernel_size=3, activation='relu')
            self.pool1 = gluon.nn.MaxPool2D(pool_size=2)
            self.conv2 = gluon.nn.Conv2D(channels=16, kernel_size=3, strides=(1,1), activation='relu')
            self.pool2 = gluon.nn.MaxPool2D(pool_size=2)
            self.dense1 = gluon.nn.Dense(units=128, activation='relu')
            self.dense2 = gluon.nn.Dense(units=10)

    def forward(self, x):
        a = self.conv1(x)
        b = self.pool1(a)
        c = self.conv2(b)
        d = self.pool2(c)
        e = self.dense1(d)
        f = self.dense2(e)
        return f

def train_block(net, train_data, eval_data=None):
    net.initialize()
    accuracy = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', optimizer_params={'learning_rate':0.1})

    epoch = 10
    for e in range(epoch):
        for i, (train_x, train_y) in enumerate(train_data):
            train_x = train_x.as_in_context(mx.cpu())#.reshape((-1, 784))

            train_y = train_y.as_in_context(mx.cpu())
            with autograd.record():
                output = net(train_x)
                one_loss = loss(output, train_y)
                one_loss.backward()
            prediction = ndarray.argmax(output, axis=1)
            accuracy.update(labels=train_y, preds=prediction)
            trainer.step(train_x.shape[0])
            curr_loss = ndarray.mean(one_loss).asscalar()
            if i%200 == 0:
                print('epoch:{}, step:{}, loss:{:.4f}, accuracy:{:.4f}'.format(e, i, curr_loss, accuracy.get()[1]))

        eval_accu = evaluate(net, eval_data)
        print("epoch:{}, evaluate accuracy:{:.2f}".format(e, eval_accu))

def evaluate(model, eval_data):
    acc = mx.metric.Accuracy()
    for eval_x, eval_y in eval_data:
        output = model(eval_x)
        prediction = ndarray.argmax(output, axis=1)
        acc.update(labels=eval_y, preds=prediction)
    return acc.get()[1]




if __name__ == "__main__":
    train_data, valid_data = load_data()

    model = GluonSeq()
    print(model)
    model.train(train_data, valid_data)

    # model = GluonBlock()
    # print(model)
    # train_block(model, train_data, valid_data)