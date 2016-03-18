# Python tutorial on neural networks with nervanasys neon
# -------------------------------------------------------
# A multilayer neural network based on OTTO Kaggle data
# Implementation in neon

# Imports
import pandas as pd
import numpy as np

# Function for normalization of inputs to a range of 0-1
def norm(x):
    return (x - x.min()) / (x.max() - x.min())

# Load training data
data = pd.read_csv("data/train.csv")
data = data.dropna(how = 'any') # remove NaN

# Store target
y_data = data['target']             # Target column
y_data = y_data.astype('category')  # Convert y to numeric
y_data = y_data.cat.codes           #  category codes
y_data = pd.get_dummies(y_data)     # make 1-hot coding

# Get the predictors
x_data = data.ix[:, 1:94] # 93 predictors (0-indexing)
x_data = norm(x_data) # Normalize predictors

# Build training and validation data
idx_train = np.random.rand(len(data)) < 0.7  # 70% training data
x_train = x_data.ix[idx_train]               # Predictor training data
x_test  = x_data.ix[~idx_train]              # Predictor test data
y_train = y_data[idx_train]                  # Target training data
y_test = y_data[~idx_train]                  # Target test data

# NEON requires np.array()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Number of rows and cols
n_row = x_data.shape[0]
n_row_train = x_train.shape[0]
n_col_x = x_data.shape[1]
n_col_y = y_data.shape[1]

# Number of inputs and classes
n_input = n_col_x
n_classes = n_col_y

# NEON Parser
from neon.util.argparser import NeonArgparser
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# Setup CPU backend
#from neon.backends import gen_backend
#gen_backend(backend='cpu', batch_size=128)

# setup training and test set iterator
from neon.data import ArrayIterator
train_set = ArrayIterator(x_train, y_train, nclass=n_classes)
test_set = ArrayIterator(x_test, y_test, nclass=n_classes)

# Initializer for weights
from neon.initializers import Gaussian
init_norm = Gaussian(loc=0.0, scale=0.01)

# Model architecture
from neon.layers import Affine
from neon.transforms import Rectlin, Softmax

# 3 layer MLP with 512, 256 and 128 neurons
layers = []
layers.append(Affine(nout=512, init=init_norm, activation=Rectlin())) # Affine = fully connected
layers.append(Affine(nout=256, init=init_norm, activation=Rectlin())) # Affine = fully connected
layers.append(Affine(nout=128, init=init_norm, activation=Rectlin())) # Affine = fully connected
layers.append(Affine(nout=n_classes, init=init_norm, activation=Softmax()))

# initialize model object
from neon.models import Model
mlp = Model(layers=layers)

# Define cost function
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# Learning
from neon.optimizers import GradientDescentMomentum
optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

# Callback function
from neon.callbacks.callbacks import Callbacks
callbacks = Callbacks(mlp, eval_set=test_set, **args.callback_args)

# Fit model
mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

# Get outputs
results = mlp.get_outputs(test_set)

# Performance
from neon.transforms import Misclassification
# evaluate the model on test_set using the misclassification metric
error = mlp.eval(test_set, metric=Misclassification())*100
print('Misclassification error = %.1f%%' % error)