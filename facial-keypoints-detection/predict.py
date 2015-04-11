import sys
import cPickle as pickle
from con import load 

X, _ = load(True)
with open('net2.pickle', 'r') as f:
    net2 = pickle.load(f)
    net2.predict(X) 
