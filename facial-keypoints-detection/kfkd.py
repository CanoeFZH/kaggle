import os 
import numpy as np
from pandas.io.parsers import read_csv 
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

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

net1 = NeuralNet(
        layers = [
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape = (None, 9216),
        hidden_num_units = 100,
        output_nonlinearity = None,
        output_num_units = 30,

        update = nesterov_momentum,
        update_learning_rate = 0.01,
        update_momentum = 0.9,

        regression = True,
        max_epochs = 400,
        verbose = 1,
        )

X, y = load()
print ("X.shape == {}, X.min == {:.3f}, X.max == {:.3f}").format(X.shape, X.min(), X.max())
print ("y.shape == {}, y.min == {:.3f}, y.max == {:.3f}").format(y.shape, y.min(), y.max())
net1.fit(X, y)

