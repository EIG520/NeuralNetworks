# Neural Networks
Implements neural networks in rust!

## FFNet
Feedforward network
allows you to make a network, and use 'gen_out' to get the outputs of the network given inputs

## TrainableFFNet
A trainable feed forward network.
Implements 'gen_out' (though it may be slower than FFNet; I haven't tested)
Implements 'train_on' which takes in a vector of inputs (an input is a vector of f32s) and a vector of expected outputs, and uses these to train then network (with the ADAM optimizer).
