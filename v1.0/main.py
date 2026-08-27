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
        }
        try:
            return funcs[str(name)]
        except KeyError:
            raise ValueError(f"Activation function '{name}' is not defined. Available: {list(funcs)}")

class NeuralNetwork:
    def __init__(self,  
                 shape=list,                 
                 label="neural_network",
                 act=Activator.sigmoid, 
                 act_deriv=Activator.sigmoid_deriv,
                 init_scale=1,
                 learning_rate=0.01,
                 init_params=True):
        
        self.label = label
        self.shape = shape
        self.depth = len(shape)
        self.act = act
        self.act_deriv = act_deriv
        self.init_scale = init_scale
        self.learning_rate = learning_rate

        self.W = [0] * self.depth
        self.B = [0] * self.depth

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
            f"Act={self.act.__name__} \n"
            f"Act_deriv={self.act_deriv.__name__} \n"
            f"Learning_rate={self.learning_rate} \n"
        )

    def forward(self, x0):
        z = [0] * self.depth
        a = [0] * self.depth
        a[0] = x0

        for i in range(1, self.depth):
            z[i] = self.W[i] @ a[i - 1] + self.B[i]
            a[i] = self.act(z[i])

        return a[-1]

    def backprop(self, x0, y):
        z = [None] * self.depth
        a = [None] * self.depth
        a[0] = x0

        for i in range(1, self.depth):
            z[i] = self.W[i] @ a[i - 1] + self.B[i]
            a[i] = self.act(z[i])

        delta = [None] * self.depth
        dB = [None] * self.depth
        dW = [None] * self.depth

        delta[-1] = 2 * (a[-1] - y) * self.act_deriv(z[-1])
        dB[-1] = np.mean(delta[-1], axis=1, keepdims=True)
        dW[-1] = delta[-1] @ a[-2].T / delta[-1].shape[1]

        for i in range(self.depth - 2, 0, -1):
            delta[i] = (self.W[i + 1].T @ delta[i + 1]) * self.act_deriv(z[i])
            dB[i] = np.mean(delta[i], axis=1, keepdims=True)
            dW[i] = delta[i] @ a[i - 1].T / delta[i].shape[1]

        for i in range(1, self.depth):
            self.W[i] -= self.learning_rate * dW[i]
            self.B[i] -= self.learning_rate * dB[i]

    def save(self, filename, folder="models"):
        import os
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)

        np.savez(path,
                 label=self.label,
                 shape=self.shape,
                 W=np.array(self.W, dtype=object),
                 B=np.array(self.B, dtype=object),
                 act=self.act.__name__,
                 act_deriv=self.act_deriv.__name__,
                 learning_rate=self.learning_rate)

    @classmethod
    def load(cls, filename, folder="models"):
        import os
        path = os.path.join(folder, filename)
        data = np.load(path, allow_pickle=True)

        label = data["label"]
        shape = data["shape"].tolist()
        W = data["W"]
        B = data["B"]
        learning_rate = data["learning_rate"]
        act = Activator.get(data["act"])
        act_deriv = Activator.get(data["act_deriv"])

        nn = cls(shape,
                 label=label,
                 act=act,
                 act_deriv=act_deriv,
                 learning_rate=learning_rate,
                 init_params=False)
        nn.W = W
        nn.B = B

        return nn
