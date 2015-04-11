import os 
import numpy as np
from pandas.io.parsers import read_csv 
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from sklearn.utils import shuffle
import theano
import cPickle as pickle

FTRAIN = 'training.csv'
FTEST = 'test.csv'

def load(test = False, cols = None):
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname)) 

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep = ' '))

    if cols:
        df = df[list(cols) + ['Image']]
    
    print df.count()
    df = df.dropna()
    X = np.vstack(df['Image'].values) / 255
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48 
        X, y = shuffle(X, y, random_state = 42)
        y = y.astype(np.float32)

    else:
        y = None

    return X, y

def load2d(test = False, cols = None):
    X, y = load(test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def float32(k):
    return np.cast['float32'](k)

class FlipBatchIterator(BatchIterator):

    flip_indices = [
            (0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16),
            (13, 17), (14, 18), (15, 19), (22, 24), (23, 25),
            ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace = False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            yb[indices, ::2] = yb[indices, ::2] * -1
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (yb[indices, b], yb[indices, a])

        return Xb, yb

class AdjustVariable(object):
    def __init__(self, name, start = 0.03, stop = 0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

net4 = NeuralNet(
        layers = [
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer), 
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer), 
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape = (None, 1, 96, 96),
        conv1_num_filters = 32, conv1_filter_size = (3, 3), pool1_ds = (2, 2),
        conv2_num_filters = 32, conv2_filter_size = (3, 3), pool2_ds = (2, 2),
        conv3_num_filters = 32, conv3_filter_size = (3, 3), pool3_ds = (2, 2),
        hidden4_num_units = 500, hidden5_num_units = 500,
        output_num_units = 30, output_nonlinearity = None,

        update_learning_rate = theano.shared(float32(0.03)),
        update_momentum = theano.shared(float32(0.9)),

        regression = True,
        batch_iterator_train = FlipBatchIterator(batch_size = 128),

        on_epoch_finished = [
            AdjustVariable('update_learning_rate', start = 0.03, stop = 0.0001),
            AdjustVariable('update_momentum', start = 0.9, stop = 0.999),
            ],
        max_epochs = 3000,
        verbose = 1,
        )

X, y = load2d()
net4.fit(X, y)

with open('net4.pickle', 'wb') as f:
    pickle.dump(net4, f, -1)

