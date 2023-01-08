# WILANNs: Width Invariant Learning Artificial Neural Networks
WILANNs is an open source PyTorch package that provides an alteration to the backpropagation algorithm, scaling pre-response activations instead of weight initializations, to achieve width invariance for the learning rate of a neural network architecture.

## How it works
The backpropagation algorithm updates the weights of a neural network based on the gradient of the loss function with respect to the weights. However, the learning rate at which the weights are updated can have a significant impact on the training process and the final performance of the network. A common issue with traditional backpropagation is that the learning rate may need to be adjusted based on the width (i.e., number of neurons) of the network, as wider networks may require a smaller learning rate to prevent divergence during training.

WILANNs addresses this issue by scaling the pre-response activations of the network instead of the weight initializations. This allows the learning rate to be invariant to the width of the network, as the activations are normalized to have the same magnitude regardless of the number of neurons in the network.