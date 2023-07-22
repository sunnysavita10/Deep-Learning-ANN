# Batch normalization

### Issues with training Deep Neural Networks 

- There are 2 major issues 1) Internal Covariate shift, 2) Vanishing Gradient

### Internal Covariate shift

- The concept of covariate shift pertains to the change that occurs in the distribution of the input to a learning system. In deep networks, this distribution can be influenced by parameters across all input layers. Consequently, even minor changes in the network can have a significant impact on its output. This effect gets magnified as the signal propagates through the network, which can result in a shift in the distribution of the inputs to internal layers. This phenomenon is known as internal covariate shift.

- When inputs are whitened (i.e., have zero mean and unit variance) and are uncorrelated, they tend to converge faster during training. However, internal covariate shift can have the opposite effect, as it introduces changes to the distribution of inputs that can slow down convergence. Therefore, to mitigate this effect, techniques like batch normalization have been developed to normalize the inputs to each layer in the network based on statistics of the current mini-batch.

### Vanishing Gradient

- Saturating non-linearities such as sigmoid or tanh are not suitable for deep networks, as the signal tends to get trapped in the saturation region as the network grows deeper. This makes it difficult for the network to learn and can result in slow convergence during training. To overcome this problem we can use the following.

- Non-linearities like ReLU which do not saturate.
- Smaller learning rates
- Careful initializations
---
### What is Normalization?

- Normalization in deep learning refers to the process of transforming the input or output of a layer in a neural network to improve its performance during training. The most common type of normalization used in deep learning is batch normalization, which normalizes the activations of a layer for each mini-batch during training.
---
### What is batch normalization?

- Batch normalization is a technique in deep learning that helps to standardize and normalize the input to each layer of a neural network by adjusting and scaling the activations. The idea behind batch normalization is to normalize the inputs to a layer to have zero mean and unit variance across each mini-batch of the training data.

### Steps involved in batch normalization

1) During training, for each mini-batch of data, compute the mean and variance of the activations of each layer. This can be done using the following formulas:

- Mean: $\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$

- Variance: $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$

- Here, $m$ is the size of the mini-batch, and $x_i$ is the activation of the $i$-th neuron in the layer.

2) Normalize the activations of each layer in the mini-batch using the following formula:

- $\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
Here, $\epsilon$ is a small constant added for numerical stability.

3) Scale and shift the normalized activations using the learned parameters $\gamma$ and $\beta$, respectively:

- $y_i = \gamma \hat{x_i} + \beta$
- The parameters $\gamma$ and $\beta$ are learned during training using backpropagation.

4) During inference, the running mean and variance of each layer are used for normalization instead of the mini-batch statistics. These running statistics are updated using a moving average of the mini-batch statistics during training.
---
### The benefits of batch normalization include:

- Improved training performance: Batch normalization reduces the internal covariate shift, which is the change in the distribution of the activations of each layer due to changes in the distribution of the inputs. This allows the network to converge faster and with more stable gradients.

- Regularization: Batch normalization acts as a form of regularization by adding noise to the activations of each layer, which can help prevent overfitting.

- Increased robustness: Batch normalization makes the network more robust to changes in the input distribution, which can help improve its generalization performance.
---
### Code example for batch normalization

```python
import tensorflow as tf

# Define a fully connected layer
fc_layer = tf.keras.layers.Dense(units=128, activation='relu')

# Add batch normalization to the layer
bn_layer = tf.keras.layers.BatchNormalization()

# Define the model with the layer and batch normalization
model = tf.keras.models.Sequential([fc_layer, bn_layer])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10)
```

- In the above code,the tf.keras.layers.BatchNormalization() layer is added after the fully connected layer to normalize the output before passing it to the activation function. The model.fit() function is then used to train the model using batch normalization.