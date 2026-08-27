import numpy as np

class NeuralNetwork:
    def __init__(
        self,  
        shape: list[int],   
        act_names: list[str], 
        loss_function_name="mse",  # Default loss function is mean squared error
        version="v2.0",             
        label="neural_network",
        learning_rate=0.01,
        init_params=True,
        init_scale=1
    ):
        """
        Initialize a neural network.

        Parameters:
            shape (list[int]): A list representing the number of neurons in each layer of the network. 
                                Should have at least two elements (input and output layers).
            act (list): A list of activation functions (one for each hidden layer).
            loss_function_deriv (function): The loss function to be used during training. Default is mean squared error.
            version (str): Version label for the network.
            label (str): Name or identifier for the neural network.
            learning_rate (float): The learning rate for weight updates during training. Must be positive.
            init_params (bool): If True, initialize weights and biases randomly. Defaults to True.
            init_scale (float): The scale for initializing weights and biases. Must be positive.

        Raises:
            ValueError: If any input is invalid (e.g., shape, act, act_deriv, loss_function_deriv).
        """
        
        # ==== Check validity of inputs:

        # 1. Validate shape: Must be a list of integers with at least two elements.
        if not isinstance(shape, list) or not all(isinstance(i, int) for i in shape):
            raise ValueError("Shape must be a list of integers.")
        if len(shape) < 2:
            raise ValueError("Shape must have at least two elements (input and output layers).")
        
        # 2. Validate activation functions: 'act' should be lists of functions.
        if not isinstance(act_names, list) or not all(isinstance(i, str) for i in act_names):
            raise ValueError("Activation functions 'act' must be a list of strings.")
        
        # Ensure the length of activation functions matches the depth of the network
        if len(act_names) != len(shape) - 1:
            raise ValueError("The length of 'act' must be one less than the length of 'shape'.")

        # 3. Validate the loss function: It should be callable (e.g., mse_loss, cross_entropy)
        if not isinstance(loss_function_name, str):
            raise ValueError("Loss function must be a string.")
        
        # 4. Validate learning_rate: It should be a positive float.
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number.")
        
        # 5. Validate init_scale: It should be a positive float or integer.
        if not isinstance(init_scale, (float, int)) or init_scale <= 0:
            raise ValueError("Initialization scale must be a positive number.")

        # Recover functions
        import modules.activation_fn as afn
        import modules.loss_fn as lfn
        
        # ==== Initialize the neural network
        # Identity
        self.version = version     
        self.label = label

        # Structure
        self.shape = shape
        self.depth = len(shape)  # Total number of layers
        self.act_names = act_names # Activation functions names
        self.act = [afn.get_actfn(name)[0] for name in act_names]  # Activation functions (act[i - 1] for layer i)
        self.act_deriv = [afn.get_actfn(name)[1] for name in act_names]  # Activation function derivatives (act_deriv[i - 1] for layer i)

        # Backpropagation variables
        self.loss_function_name = loss_function_name
        self.loss_function_deriv = lfn.get_lossfn(loss_function_name)[1]
        self.learning_rate = learning_rate

        # Weights and biases initialization
        self.W = [None] * (self.depth - 1)  # Weights (W[i - 1] for i)
        self.B = [None] * (self.depth - 1)  # Biases (B[i - 1] for i)
        
        # Initialize random weights and biases if requested
        self.init_scale = init_scale
        if init_params:
            for i in range(0, self.depth - 1):
                # Initialize weights and biases with a uniform distribution within [-init_scale, init_scale]
                self.W[i] = np.random.uniform(-self.init_scale, 
                                              self.init_scale, 
                                              (shape[i + 1], shape[i]))
                self.B[i] = np.random.uniform(-self.init_scale, 
                                              self.init_scale, 
                                              (shape[i + 1], 1))

    def __str__(self):
        """
        Return a string representation of the neural network with key parameters.

        Returns:
            str: A formatted string summarizing the key properties of the network.
        """
        return (
            f"Neural Network '{self.label}' ({self.version})\n"
            f"Shape (layers): {self.shape}\n"
            f"Activation functions: {', '.join([act.__name__ for act in self.act])}\n"
            f"Loss function: {self.loss_function_deriv.__name__}\n"
            f"Learning rate: {self.learning_rate}\n"
            f"Initialization scale: {self.init_scale}\n"
            f"Depth (layers): {self.depth}"
        )
    
    def forward(self, x, backprop_return=False):
        """
        Perform a forward pass through the neural network.

        This method computes the activations for each layer in the network, given an input.
        It calculates the weighted sum of inputs (`z`) and applies the activation function (`a`) for each layer.

        Parameters:
            x (ndarray): The input data to the neural network. Should have the shape (input_size, 1) for a single sample.
            backprop_return (bool): If True, the method returns both the activations (`a`) and the weighted sums (`z`) 
                                    for use in backpropagation. Defaults to False, in which case only the output is returned.

        Returns:
            ndarray: The output of the neural network (final layer activations).
            If `backprop_return` is True, it also returns the activations and weighted sums for all layers 
            in the form of two lists: `a` (activations) and `z` (weighted sums).

        Raises:
            ValueError: If the input shape is inconsistent with the network's expected input size.
        """
        
        # ==== Initialize lists 
        z = [None] * self.depth  # Weighted sums (z[i] is for layer i)
        a = [None] * self.depth  # Activations (a[i] is for layer i)
        a[0] = x  # First layer activations

        # ==== Forward pass through each layer (except input layer)
        for i in range(1, self.depth):

            # Compute the weighted sum (z) for the current layer
            z[i] = self.W[i - 1] @ a[i - 1] + self.B[i - 1]  # Linear transformation (W * a + B)
            
            # Apply the activation function for the current layer
            a[i] = self.act[i - 1](z[i])
        
        # The final output is the activation of the last layer
        y = a[-1]

        # If backpropagation values are requested, return both activations and weighted sums.
        if backprop_return:
            return y, a, z  # Return output, activations for each layer, and weighted sums
        else:
            return y  # Return only the output (final layer activations)


    def backprop(self, x, y_true):
        """
        Perform backpropagation and update weights and biases.

        Parameters:
            x (ndarray): Input data of shape (input_size, batch_size).
            y_true (ndarray): True labels of shape (output_size, batch_size).

        This method:
        - Performs a forward pass to compute activations.
        - Computes the gradient of the loss with respect to weights and biases.
        - Updates the weights and biases using gradient descent.

        Assumptions:
        - Activation derivatives are elementwise (except softmax, where the derivative simplifies).
        - Loss function accepts predicted output, true output, pre-activation (z), and activation derivative.
        """

        # === Forward pass (to get activations and weighted inputs)
        y_pred, a, z = self.forward(x, backprop_return=True)

        # === Initialize gradient holders for each layer (0-based indexing)
        delta = [None] * (self.depth - 1)  # Error terms
        dB = [None] * (self.depth - 1)     # Gradient wrt biases
        dW = [None] * (self.depth - 1)     # Gradient wrt weights

        # === Output layer delta
        delta[-1] = self.loss_function_deriv(y_pred, y_true, z[-1], self.act_deriv[-1])
        dB[-1] = np.mean(delta[-1], axis=1, keepdims=True)  # Mean across batch
        dW[-1] = delta[-1] @ a[-2].T / delta[-1].shape[1]    # Outer product averaged across batch

        # === Backpropagate to hidden layers (in reverse)
        for i in reversed(range(self.depth - 2)):  # from L-2 to 0
            delta[i] = (self.W[i + 1].T @ delta[i + 1]) * self.act_deriv[i](z[i + 1])
            dB[i] = np.mean(delta[i], axis=1, keepdims=True)
            dW[i] = delta[i] @ a[i].T / delta[i].shape[1]

        # === Gradient descent update
        for i in range(self.depth - 1):
            self.W[i] -= self.learning_rate * dW[i]
            self.B[i] -= self.learning_rate * dB[i]

    def save(self, filename, folder="models"):
        """
        Saves the neural network's configuration and parameters to a .npz file.

        Parameters:
            filename (str): Name of the file to save (e.g., 'model1.npz').
            folder (str): Directory in which to save the model. Default is 'models'.

        What gets saved:
            - version: Custom version identifier for this network.
            - label: Descriptive label for identification.
            - shape: Layer sizes of the network.
            - W: Weights for each layer (dtype=object).
            - B: Biases for each layer (dtype=object).
            - act: List of activation function names (by __name__).
            - act_deriv: List of derivative function names (by __name__).
            - loss_function_deriv: Name of the loss function used.
            - learning_rate: Scalar learning rate used in training.
        """
        import os

        # Ensure directory exists
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)


        # Save everything into a compressed .npz archive
        np.savez(path,
                version=self.version,
                label=self.label,
                shape=self.shape,
                W=np.array(self.W, dtype=object),
                B=np.array(self.B, dtype=object),
                act_names=self.act_names,
                loss_function_name=self.loss_function_name,
                learning_rate=self.learning_rate)
        

    @classmethod
    def load(cls, filename, folder="models"):
        """
        Loads a saved neural network from a .npz file and reconstructs the object.

        Parameters:
            filename (str): The name of the saved file (e.g., 'model1.npz').
            folder (str): The directory where the file is stored. Default is 'models'.

        Returns:
            NeuralNetwork: An instance of the class with loaded parameters and configuration.
        
        Expected fields in the file:
            - version: Version string of the model.
            - label: Identifier or label for the model.
            - shape: List of layer sizes.
            - W: Weights (dtype=object).
            - B: Biases (dtype=object).
            - act_names: List of activation function names.
            - loss_function_name: Name of the loss function.
            - learning_rate: Training learning rate.
        """
        import os

        # Load the model data from the .npz file
        path = os.path.join(folder, filename)
        data = np.load(path, allow_pickle=True)

        # Extract stored parameters
        version = str(data["version"])
        label = str(data["label"])
        shape = data["shape"].tolist()   
        W = data["W"]   
        B = data["B"]   
        learning_rate = float(data["learning_rate"])   
        act_names = data["act_names"].tolist()   
        loss_function_name = str(data["loss_function_name"])   

        # Construct and return the network
        nn = cls(shape=shape,
                act_names=act_names,
                loss_function_name=loss_function_name,
                version=version,
                label=label,
                learning_rate=learning_rate,
                init_params=False)

        nn.W = W.tolist()
        nn.B = B.tolist()

        return nn

