# nnet-mnist

There are currently two implementations: `nnet1.c` uses online training (i.e.
it computes the gradient for one input at a time) and `nnet2.c` which uses
stochastic gradient descent (i.e. it averages the gradient over a batch of
inputs).

To run the examples, first download the
[MNIST data](http://yann.lecun.com/exdb/mnist/) and unpack it to the `mnist/`
folder.  Then type `make nnet1 && ./nnet1` or `make nnet2 && ./nnet2` to
compile and run either example.  Both will train a network and then print how
many correct predictions they made on the test data.

The theory for these classifiers is covered in the first chapter of the free
online book
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com).
