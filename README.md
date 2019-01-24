This project explores the effect of physicality constraints on neural networks. This is a part of my synthesis project.

Essentially each neuron is given a physical co-ordinate and trained by adding this to the loss function. For more insights please visit: [project wiki](https://github.com/LousyLory/Physical_Constraints_NN/wiki)

To start training please run `python main.py`

----

<h2>Project details</h2>

----

Welcome to the Physical_Constraints_NN wiki!

This project explores the effects of adding physical effects to a neural network's training. These effects for now are of two kinds:

(a) Spatial constraint,
(b) Energy constraint.

----

<h3>Original network</h3>

The original output of the initial layer before any non-linearity is given as:

$f(x,W,b) = x'*W' + b.$

The fully connected weights of a trained network is shown below. Notice the almost duplicate filters. This occurs because the network is over-parameterized. 

<img src="https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/model1_modified_0.0W1.png" />

![image1](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/model1_modified_0.0W1.png)

The training and loss plots are shown below:

![image2](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/train_accuracy_plot_modified_0.0.png)

----

<h3>Spatial constraint</h3>

In this section, we shall be looking into using *spatial constraint* to induce a *bias* in learning. Several methods can be used to perform such operation. Following are the theory and associated experiments performed:

(a) Each filter is assigned a position in the co-ordinate system. We construct this for now using uniform spacing between the filters centered at (0,0). For example let us have 5 filters. The co-ordinates of each filter will be (0,-1), (-1,0), (0,0), (1,0), (1,1). Define L_1 as the *hamming distance* of each filter from (0,0). Define the loss function as: 

$f(x,W,b) = x'*W' + b + \alpha * L_1,$

where $\alpha \in [0,1]$. Thus only the forward pass is affected. The results for comparison are:

1. For $\alpha$ == 0.5, the network starts to show a bias towards regional learning. Although one might notice the introduction of some random filter weights. The resultant fully connected filters are: ![image3](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/model1_modified_0.5W1.png)

2. For $\alpha$ == 0.9, the network shows strong bias towards regional grouping of filters. The number of random filter weights have increased considerably. The resultant fully connected filters are: ![image4](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/model1_modified_0.9W1.png)

The plots for training and validation for each (1 and 2) are shown in the following figure respectively:

1. ![image5](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/train_accuracy_plot_modified_0.5.png)

2. ![image6](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/train_accuracy_plot_modified_0.9.png)

(b) Here we switch the distance function from $L_1$ to $L_2$ (Euclidean). The results isn't tat interesting as we can see that the zonal clustering of the filters do not exist. Results for different mixing parameters are shown below:

1. Using $\alpha == 0.5$: ![image7](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/model1_modified_L2_0.5W1.png)

2. Using $\alpha == 0.9$: ![image8](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/model1_modified_L2_0.9W1.png)

Train and Loss plots for the above are shown below:

1. ![image9](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/train_accuracy_plot_modified_L2_0.5.png)

2. ![image10](https://github.com/LousyLory/Physical_Constraints_NN/blob/master/outputs/train_accuracy_plot_modified_L2_0.9.png)
