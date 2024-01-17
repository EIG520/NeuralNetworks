# Neural Networks
Implements neural networks in rust!

## FFNet
Feedforward network
allows you to make a network, and use 'gen_out' to get the outputs of the network given inputs

## TrainableFFNet
A trainable feed forward network.
Implements 'gen_out' (though it may be slower than FFNet: I haven't tested)
Implements 'train_one_case' which lets you supply input, output, and a learning rate to train on once
