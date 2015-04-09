import os 
import numpy as np
from pandas.io.parsers import read_csv 
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as pyplot

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
        max_epochs = 1,
        verbose = 1,
        )

X, y = load()
print ('X.shape == {}, X.min == {:.3f}, X.max == {:.3f}').format(X.shape, X.min(), X.max())
print ('y.shape == {}, y.min == {:.3f}, y.max == {:.3f}').format(y.shape, y.min(), y.max())
net1.fit(X, y)

train_loss = np.array([i['train_loss'] for i in net1.train_history_])
valid_loss = np.array([i['valid_loss'] for i in net1.train_history_])
pyplot.plot(train_loss, linewidth = 3, label = 'train')
pyplot.plot(valid_loss, linewidth = 3, label = 'valid')
pyplot.grid()
pyplot.legend()
pyplot.xlabel('epoch')
pyplot.ylable('loss')
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale('log')
pyplot.show


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap = 'gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker = 'x', s = 10)

X, _ = load(test = True)
y_pred = net1.predict(X)

fig = pyplot.figure(figsize(6, 6))
fig.subplit_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks = [], yticks = [])
    plot_sample(X[i], y_pred[i], ax)

pyplot.show()
