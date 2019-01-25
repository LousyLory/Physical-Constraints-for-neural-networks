import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from src.classifiers.fc_net import *
from src.data_utils import get_CIFAR10_data
from src.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from src.solver import Solver
from src.canonicalize import *
from display_funcs import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

########################################################################################
# define relatie error
def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
########################################################################################

########################################################################################
# Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()
for k, v in data.iteritems():
	print '%s: ' % k, v.shape
########################################################################################

########################################################################################
# format data accordingly
X_train = data['X_train']
y_train = data['y_train']

data1 = {
		'X_train': X_train, 'y_train' : y_train,
		'X_val': data['X_val'], 'y_val' : data['y_val'],
		'X_test': data['X_test'], 'y_test' : data['y_test'],
		}
########################################################################################

########################################################################################
# finally train the model here
# here I use the same parameters I got before, seems like they perform moderately fine
# also this network employs fc net with spatial constraint
mp = float(sys.argv[1])
model1 = FullyConnectedNet([100, 100], weight_scale=0.003, use_batchnorm = False, \
									reg=0.6, spatial_update=False, \
									spatial_update_type='L1', mixing_param=mp,\
									correlate_update=True)
solver1 = Solver(model1, data1,
						print_every=data1['X_train'].shape[0], num_epochs=50, batch_size=100,
						update_rule='sgd',
						optim_config={
						  'learning_rate': 0.03,
						},verbose=True, lr_decay = 0.9,
				 )
solver1.train()
pass
########################################################################################

########################################################################################
# plot stuff
plot_train_loss(solver1, 'train_accuracy_plot_modified_xx_'+str(mp))
save_net_weights(model1, 'model1_modified_xx_'+str(mp))
########################################################################################
