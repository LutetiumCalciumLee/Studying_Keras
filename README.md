# Keras Theory

## 1. Machine Learning Foundations and High-Level Deep Learning

### TensorFlow (Low-Level Tensor Operations)
- **Tensor**: The basic data structure that includes both constants and variables, and is fundamentally constant in nature.
- **Tensor Operations**: Supports mathematical operations such as addition, relu, matmul, etc.
- **Backpropagation**: The core method for calculating gradients during neural network training.

### Keras (High-Level Deep Learning API)
- **Model Layers**: The fundamental building blocks of deep learning models.
- **Loss Function**: The criterion for measuring model performance.
- **Optimizer**: Determines how weights are updated.
- **Metrics**: Indicators for evaluating model performance.
- **Training Loop**: Performs mini-batch stochastic gradient descent.

## 2. Tensor Operations in TensorFlow

### Constant Tensors
```python
# Main tensor creation functions
tf.ones(shape=(2, 1))      # Tensor filled with ones
tf.zeros(shape=(2, 1))     # Tensor filled with zeros
tf.random.normal()         # Random numbers from normal distribution
tf.random.uniform()        # Random numbers from uniform distribution
```

### Variable Tensors
- **tf.Variable**: A class for managing mutable state.
- **assign()**: Assigns a value to the variable.
- **assign_add()**: Adds to the variable value.
- **assign_sub()**: Subtracts from the variable value.

### Gradient Calculation
- **tf.GradientTape()**: General tool for gradient calculation.
  - Automatically tracks only trainable variable tensors.
  - For constant tensors, manual tracking is needed with tape.watch().
  - Supports higher-order derivatives.

## 3. Keras Layer Concept

### Layer Properties
- **Definition**: A data processing module that takes one or more tensors as input and outputs one or more tensors.
- **Weights**: One or more tensors learned through stochastic gradient descent.
- **Tensor Format**: Each layer supports different tensor shapes:
  - Rank-2 tensor (samples, features): Dense (fully connected) layers
  - Rank-3 tensor (samples, timesteps, features): Recurrent layers, 1D convolutional layers

### Keras Layer Class Example
```python
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        # Define layer parameters
    
    def build(self, input_shape):
        # Create weights (executed automatically on first call)
    
    def call(self, inputs):
        # Define forward pass computation
```

## 4. Keras Model Construction Methods

### 4.1 Sequential Model
- **Characteristics**: The simplest way, stacking layers sequentially.
- **Limitations**: Only supports single input, single output, and strictly sequential structures.
- **Use Cases**: Basic neural network structures.

```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

### 4.2 Functional API
- **Characteristics**: Can handle graph-like model structures.
- **Advantages**: A good balance between usability and flexibility.
- **Use Cases**: Multi-input/output, non-linear structures.

```python
inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)
```

### 4.3 Model Subclassing
- **Characteristics**: Low-level method for building from scratch.
- **Advantages**: Complete control over all details.
- **Disadvantages**: 
  - Cannot output layer connection structure with summary()
  - Cannot visualize structure with plot_model()
  - Cannot use layers externally

## 5. Model Training Process

### 5.1 Compile
Three key elements to decide for training:

1. **Loss Function**: The value to minimize during training
   - CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy
2. **Optimizer**: Determines how the network is updated
   - SGD, RMSprop, Adam, Adagrad
3. **Metrics**: Success indicators to monitor during training and validation
   - CategoricalAccuracy, SparseCategoricalAccuracy, BinaryAccuracy

### 5.2 Training (Fit)
```python
model.fit(
    inputs,
    targets,
    epochs=5,           # Number of training epochs
    batch_size=128,     # Batch size
    validation_data=(val_inputs, val_targets)  # Validation data
)
```

## 6. Major Application Examples

### 6.1 Binary Classification (IMDB Movie Reviews)
- **Data**: 50,000 movie reviews (positive/negative)
- **Preprocessing**: Vectorized using multi-hot encoding
- **Model Structure**: Dense-Dense-Dense with sigmoid output
- **Loss Function**: binary_crossentropy

### 6.2 Multi-Class Classification (Reuters News Classification)
- **Data**: News articles from 46 topics
- **Preprocessing**: One-hot encoding
- **Model Structure**: Dense-Dense-Dense with softmax output
- **Loss Function**: categorical_crossentropy

### 6.3 Regression (Boston Housing Price Prediction)
- **Data**: 404 training samples, 102 test samples
- **Preprocessing**: Data normalization required
- **Validation Method**: k-fold cross-validation
- **Loss Function**: mean_squared_error

## 7. Advanced Features

### 7.1 Callbacks
Tools for monitoring and controlling the model state during training:
- **ModelCheckpoint**: Save model weights
- **EarlyStopping**: Early stopping
- **LearningRateScheduler**: Dynamic learning rate adjustment

### 7.2 Custom Metrics
```python
class RootMeanSquaredError(keras.metrics.Metric):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # State update logic
    
    def result(self):
        # Final result calculation
    
    def reset_state(self):
        # State initialization
```

### 7.3 Multi-Input/Output Models
- **Input**: Various types of data (text, categorical, etc.)
- **Output**: Various types of predictions (priority, classification, etc.)
- **Implementation**: Use the Functional API
