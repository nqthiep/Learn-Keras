###Dense
Dense implements the operation: output = activation(dot(input, kernel) + bias)

Params:
* Activation is the element-wise activation function passed as the activation argument.
    + softmax
    + elu
    + softplus
    + softsign
    + relu
    + tanh
    + sigmoid
    + hard_sigmoid
    + linear
* Kernel is a weights matrix created by the layer
* Bias is a bias vector created by the layer (only applicable if use_bias is True).

###Activation
Activation Applies an activation function to an output.

Params:
* activation: name of activation function to use
* Input shape: Use the keyword argument input_shape when using this layer as the first layer in a model.
* Output shape: Use the keyword argument output_shape when using this layer as the first layer in a model.

###Dropout
Applies Dropout to the input.

Dropout consists in randomly setting a fraction p of input units to 0 at each update during training time, which helps prevent overfitting.

