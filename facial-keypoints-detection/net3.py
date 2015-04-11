import os 
import numpy as np
from pandas.io.parsers import read_csv 
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from sklearn.utils import shuffle
import cPickle as pickle

FTRAIN = 'training.csv'
FTEST = 'test.csv'

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

net3 = NeuralNet(
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

        update_learning_rate = 0.01,
        update_momentum = 0.9,

        regression = True,
        batch_iterator_train = FlipBatchIterator(batch_size = 12),
        max_epochs = 3000,
        verbose = 1,
        )

X, y = load2d()
net3.fit(X, y)

with open('net3.pickle', 'wb') as f:
    pickle.dump(net3, f, -1)

