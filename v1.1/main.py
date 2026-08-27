import numpy as np

class Activator:
    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoid_deriv(x): s = Activator.sigmoid(x); return s * (1 - s)

    @staticmethod
    def tanh(x): return np.tanh(x)
    @staticmethod
    def tanh_deriv(x): return 1 - np.tanh(x)**2

    @staticmethod
    def scaled_tanh(x): return 1.2 * np.tanh(x)
    @staticmethod
    def scaled_tanh_deriv(x): return 1.2 * Activator.tanh_deriv(x)

    @staticmethod
    def linear(x): return x
    @staticmethod
    def linear_deriv(x): return np.ones_like(x)

    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def relu_deriv(x): return (x > 0).astype(float)

    @staticmethod                                       
    def softmax(z):
        exp_z = np.exp(z - np.max(z))  # Subtract max(z) for numerical stability
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)    
    @staticmethod
    def softmax_deriv(z):
        return None

    @staticmethod
    def get(name):
        funcs = {
            "sigmoid": Activator.sigmoid,
            "sigmoid_deriv": Activator.sigmoid_deriv,
            "tanh": Activator.tanh,
            "tanh_deriv": Activator.tanh_deriv,
            "scaled_tanh": Activator.scaled_tanh,
            "scaled_tanh_deriv": Activator.scaled_tanh_deriv,
            "linear": Activator.linear,
            "linear_deriv": Activator.linear_deriv,
            "relu": Activator.relu,
            "relu_deriv": Activator.relu_deriv,
            "softmax": Activator.softmax,
            "softmax_deriv": Activator.softmax_deriv,

            "mse": Activator.mse,
            "cross_entropy": Activator.cross_entropy
        }
        try:
            return funcs[str(name)]
        except KeyError:
            raise ValueError(f"Activation function '{name}' is not defined. Available: {list(funcs)}")

    @staticmethod
    def mse(y_pred, y_true, z, act_deriv):
        return 2 * (y_pred - y_true) * act_deriv(z)
    @staticmethod
    def cross_entropy(y_pred, y_true, z=None, act_deriv=None):
        """
        Cross-entropy loss derivative w.r.t. pre-activation input z.
        Assumes softmax activation in output layer and one-hot y_true.
        
        If softmax is used, the gradient simplifies to:
        grad = y_pred - y_true
        """
        return y_pred - y_true

class NeuralNetwork:
    def __init__(self,  
                 shape,    
                 version="v1.1",             
                 label="neural_network",
                 act=Activator.sigmoid, 
                 act_deriv=Activator.sigmoid_deriv,
                 init_scale=1,
                 learning_rate=0.01,
                 loss_function=Activator.mse,
                 init_params=True):
        
        """
        Index conventions: \n
        a, z: starts at 0  \n
        W, B, act, act_deriv: starts at 1 \n
        """
        
        self.version = version
        
        self.label = label
        self.shape = shape
        self.depth = len(shape)
        self.init_scale = init_scale
        self.learning_rate = learning_rate
        self.loss_function = loss_function

        if isinstance(act, list):
            self.act = act
        else:
            self.act = [act] * self.depth
    
        if isinstance(act_deriv, list):
            self.act_deriv = act_deriv
        else:
            self.act_deriv = [act_deriv] * self.depth

        self.W = [None] * self.depth 
        self.B = [None] * self.depth 

        if init_params:
            for i in range(1, self.depth):
                self.W[i] = np.random.uniform(-self.init_scale, 
                                              self.init_scale, 
                                              (shape[i], shape[i - 1]))
                self.B[i] = np.random.uniform(-self.init_scale, 
                                              self.init_scale, 
                                              (shape[i], 1))

    def __str__(self):
        return (
            f"Label={self.label} \n"
            f"Shape={self.shape} \n"
            f"Act={self.act[1].__name__} \n"
            f"Act_deriv={self.act_deriv[1].__name__} \n"
            f"Learning_rate={self.learning_rate} \n"
        )

    def forward(self, x0):
        z = [None] * self.depth
        a = [None] * self.depth
        a[0] = x0

        for i in range(1, self.depth):
            z[i] = self.W[i] @ a[i - 1] + self.B[i]
            a[i] = self.act[i](z[i])

        return a[-1]

    def backprop(self, x0, y):
        z = [None] * self.depth
        a = [None] * self.depth
        a[0] = x0

        for i in range(1, self.depth):
            z[i] = self.W[i] @ a[i - 1] + self.B[i]
            a[i] = self.act[i](z[i])

        delta = [None] * self.depth
        dB = [None] * self.depth
        dW = [None] * self.depth

        delta[-1] = self.loss_function(a[-1], y, z[-1], self.act_deriv[-1])
        dB[-1] = np.mean(delta[-1], axis=1, keepdims=True)
        dW[-1] = delta[-1] @ a[-2].T / delta[-1].shape[1]

        for i in range(self.depth - 2, 0, -1):
            delta[i] = (self.W[i + 1].T @ delta[i + 1]) * self.act_deriv[i](z[i])
            dB[i] = np.mean(delta[i], axis=1, keepdims=True)
            dW[i] = delta[i] @ a[i - 1].T / delta[i].shape[1]

        for i in range(1, self.depth):
            self.W[i] -= self.learning_rate * dW[i]
            self.B[i] -= self.learning_rate * dB[i]

    
    @classmethod
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
            - loss_function: Name of the loss function used.
            - learning_rate: Scalar learning rate used in training.
        """
        import os

        # Ensure directory exists
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)

        # Retrieve function names for saving
        act_names = [fn.__name__ for fn in self.act]
        act_deriv_names = [fn.__name__ for fn in self.act_deriv]

        # Save everything into a compressed .npz archive
        np.savez(path,
                version=self.version,
                label=self.label,
                shape=self.shape,
                W=np.array(self.W, dtype=object),
                B=np.array(self.B, dtype=object),
                act=act_names,
                act_deriv=act_deriv_names,
                loss_function=self.loss_function.__name__,
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
            - act: List of activation function names.
            - act_deriv: List of activation derivative function names.
            - loss_function: Name of the loss function.
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

        # Recover functions
        import activation_fn as afn
        import loss_fn as lfn

        act_names = data["act"].tolist()
        act_deriv_names = data["act_deriv"].tolist()

        act = [afn.get_actfn(name)[0] for name in act_names]
        act_deriv = [afn.get_actfn(name)[1] for name in act_deriv_names]  # Or re-use same dict if unified

        loss_function = lfn.get_lossfn(str(data["loss_function"]))

        # Construct and return the network
        nn = cls(shape=shape,
                act=act,
                act_deriv=act_deriv,
                loss_function=loss_function,
                version=version,
                label=label,
                learning_rate=learning_rate,
                init_params=False)

        nn.W = W.tolist()
        nn.B = B.tolist()

        return nn

