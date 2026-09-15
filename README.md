<details>
<summary>ENG (English Version)</summary>

# Keras
This repository contains my notes on TensorFlow and Keras, covering the fundamental concepts of deep learning model construction, training, evaluation, and practical applications.

## 1. TensorFlow and Keras Overview

### TensorFlow

TensorFlow provides low-level operations for numerical computation and deep learning.

* **Tensor**: A multidimensional data structure used to represent numerical data.
* **Tensor Operations**: Mathematical operations such as `add`, `matmul`, and activation functions.
* **Automatic Differentiation**: Computes gradients required for neural network training.
* **`tf.Variable`**: Stores mutable values such as trainable model parameters.

### Keras

Keras is a high-level deep learning API for building and training neural networks.

* **Layers**: Fundamental building blocks of neural networks.
* **Models**: Organize layers into trainable structures.
* **Loss Functions**: Measure prediction error.
* **Optimizers**: Update model parameters using gradients.
* **Metrics**: Evaluate model performance.
* **Callbacks**: Control and monitor the training process.

---

## 2. Tensor Operations

### Creating Tensors

```python
tf.ones(shape=(2, 1))
tf.zeros(shape=(2, 1))
tf.random.normal(shape=(2, 2))
tf.random.uniform(shape=(2, 2))
```

### Variables

```python
x = tf.Variable(initial_value=3.0)

x.assign(5.0)
x.assign_add(1.0)
x.assign_sub(1.0)
```

* `tf.Variable` is mainly used for values that need to change during training.
* Neural network weights are typically stored as variables.

### Automatic Differentiation

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

gradient = tape.gradient(y, x)
```

* **`tf.GradientTape()`** records operations required for gradient calculation.
* Trainable variables are automatically tracked.
* Constant tensors can be manually tracked using `tape.watch()`.
* Nested gradient tapes can be used for higher-order derivatives.

---

## 3. Keras Layers

A layer is a data-processing component that receives tensors and returns transformed tensors.

### Common Layer Types

| Input Type                           | Typical Layer  |
| ------------------------------------ | -------------- |
| `(samples, features)`                | `Dense`        |
| `(samples, timesteps, features)`     | RNN / `Conv1D` |
| `(samples, height, width, channels)` | `Conv2D`       |

### Custom Layer

```python
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros"
        )

    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        return output
```

The main components of a custom layer are:

* **`__init__()`**: Defines layer configuration.
* **`build()`**: Creates trainable parameters.
* **`call()`**: Defines the forward computation.

---

## 4. Keras Model Construction

Keras provides three main approaches for building models.

| Method                | Characteristics                              | Suitable For                                 |
| --------------------- | -------------------------------------------- | -------------------------------------------- |
| **Sequential API**    | Layers are connected sequentially            | Simple neural networks                       |
| **Functional API**    | Defines models as computational graphs       | Multi-input/output and complex architectures |
| **Model Subclassing** | Model behavior is defined directly in Python | Highly customized models                     |

### 4.1 Sequential API

```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

Advantages:

* Simple and intuitive
* Easy to build and inspect

Limitations:

* Best suited for linear layer stacks
* Not ideal for complex branching architectures

### 4.2 Functional API

```python
inputs = keras.Input(shape=(3,))
x = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

Advantages:

* Supports multiple inputs and outputs
* Supports shared layers
* Supports non-linear computational graphs
* Easy to visualize model architecture

### 4.3 Model Subclassing

```python
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
```

Advantages:

* Maximum flexibility
* Suitable for custom forward-pass logic

Limitations:

* Model structure is less explicit than with the Functional API
* Serialization and visualization may require additional consideration
* More implementation responsibility is placed on the developer

---

## 5. Model Compilation and Training

### Compile

Before training, three main components are configured.

| Component         | Purpose                    | Examples                                                            |
| ----------------- | -------------------------- | ------------------------------------------------------------------- |
| **Loss Function** | Measures prediction error  | `BinaryCrossentropy`, `CategoricalCrossentropy`, `MeanSquaredError` |
| **Optimizer**     | Updates model parameters   | `SGD`, `RMSprop`, `Adam`, `Adagrad`                                 |
| **Metrics**       | Measures model performance | `Accuracy`, `Precision`, `Recall`, `MAE`                            |

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Training

```python
history = model.fit(
    inputs,
    targets,
    epochs=5,
    batch_size=128,
    validation_data=(val_inputs, val_targets)
)
```

Important training parameters:

* **Epoch**: One complete pass through the training dataset.
* **Batch Size**: Number of samples processed before updating model parameters.
* **Validation Data**: Data used to evaluate generalization during training.

---

## 6. Major Machine Learning Tasks

### Binary Classification

Example: IMDB movie review sentiment classification

* **Output Layer**: `Dense(1, activation="sigmoid")`
* **Loss Function**: `binary_crossentropy`
* **Purpose**: Predict one of two classes.

### Multi-Class Classification

Example: Reuters news topic classification

* **Output Layer**: `Dense(num_classes, activation="softmax")`
* **Loss Function**: `categorical_crossentropy` or `sparse_categorical_crossentropy`
* **Purpose**: Predict one class among multiple categories.

### Regression

Example: Housing price prediction

* **Output Layer**: Usually a single neuron without an activation function.
* **Loss Function**: `mean_squared_error`
* **Metrics**: `mean_absolute_error`
* **Preprocessing**: Feature normalization is often important.

---

## 7. Validation and Generalization

Model performance should be evaluated using data that was not directly used for training.

### Validation Methods

* **Hold-out Validation**
* **K-Fold Cross-Validation**
* **Training / Validation / Test Split**

### Common Problems

* **Overfitting**: Good training performance but poor validation performance.
* **Underfitting**: Poor performance on both training and validation data.

Common approaches for improving generalization include:

* Regularization
* Dropout
* Early stopping
* Data augmentation
* Increasing training data

---

## 8. Callbacks

Callbacks allow additional operations to be performed during model training.

### Common Callbacks

* **`ModelCheckpoint`**: Saves the model or model weights during training.
* **`EarlyStopping`**: Stops training when performance no longer improves.
* **`LearningRateScheduler`**: Dynamically changes the learning rate.
* **`ReduceLROnPlateau`**: Reduces the learning rate when a monitored metric stops improving.

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3
    )
]
```

---

## 9. Custom Metrics

Custom metrics can be implemented by extending `keras.metrics.Metric`.

```python
class RootMeanSquaredError(keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        pass

    def reset_state(self):
        pass
```

The main methods are:

* **`update_state()`**: Updates internal metric states.
* **`result()`**: Calculates the final metric value.
* **`reset_state()`**: Resets stored states.

---

## 10. Multi-Input and Multi-Output Models

The Functional API can be used to construct models that process multiple types of data or generate multiple predictions.

Examples:

* Text + numerical features
* Image + metadata
* Classification + regression outputs

```python
model = keras.Model(
    inputs=[input_a, input_b],
    outputs=[output_a, output_b]
)
```

## 11. Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are commonly used for image processing and computer vision tasks.

### Conv2D

`Conv2D` applies convolutional filters to an image to extract spatial features such as edges, textures, and patterns.

```python
layers.Conv2D(
    32,
    kernel_size=(3, 3),
    activation="relu"
)
```


This structure is useful when a single model needs to learn several related tasks or combine heterogeneous data sources.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 케라스
이 Repository에는 딥러닝 모델의 구성, 학습, 평가 및 활용에 필요한 TensorFlow와 Keras의 주요 개념을 정리했습니다.

## 1. TensorFlow와 Keras

### TensorFlow

TensorFlow는 수치 연산과 딥러닝을 위한 저수준 연산 기능을 제공합니다.

* **텐서(Tensor)**: 수치 데이터를 표현하는 다차원 데이터 구조
* **텐서 연산**: `add`, `matmul`, 활성화 함수 등의 수학적 연산
* **자동 미분**: 신경망 학습에 필요한 그래디언트를 자동으로 계산
* **`tf.Variable`**: 학습 가능한 모델 파라미터처럼 변경 가능한 값을 저장

### Keras

Keras는 신경망 모델을 구성하고 학습하기 위한 고수준 딥러닝 API입니다.

* **Layer**: 신경망을 구성하는 기본 단위
* **Model**: 여러 레이어를 하나의 학습 가능한 구조로 구성
* **Loss Function**: 예측 오차 측정
* **Optimizer**: 그래디언트를 이용하여 모델 파라미터 업데이트
* **Metric**: 모델 성능 평가
* **Callback**: 학습 과정 모니터링 및 제어

---

## 2. Tensor 연산

### Tensor 생성

```python
tf.ones(shape=(2, 1))
tf.zeros(shape=(2, 1))
tf.random.normal(shape=(2, 2))
tf.random.uniform(shape=(2, 2))
```

### Variable

```python
x = tf.Variable(initial_value=3.0)

x.assign(5.0)
x.assign_add(1.0)
x.assign_sub(1.0)
```

* `tf.Variable`은 학습 과정에서 변경되어야 하는 값을 저장할 때 사용
* 신경망의 가중치와 같은 학습 파라미터가 대표적인 예

### 자동 미분

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

gradient = tape.gradient(y, x)
```

* **`tf.GradientTape()`**: 그래디언트 계산에 필요한 연산을 기록
* 학습 가능한 변수는 자동으로 추적
* 상수 텐서는 `tape.watch()`를 이용하여 직접 추적 가능
* 중첩하여 사용하면 고차 도함수 계산 가능

---

## 3. Keras Layer

Layer는 하나 이상의 텐서를 입력받아 새로운 텐서를 출력하는 데이터 처리 단위입니다.

### 대표적인 Layer

| 입력 형태                                | 대표 Layer       |
| ------------------------------------ | -------------- |
| `(samples, features)`                | `Dense`        |
| `(samples, timesteps, features)`     | RNN / `Conv1D` |
| `(samples, height, width, channels)` | `Conv2D`       |

### 사용자 정의 Layer

```python
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros"
        )

    def call(self, inputs):
        output = tf.matmul(inputs, self.w) + self.b
        return output
```

주요 메서드:

* **`__init__()`**: Layer 설정 정의
* **`build()`**: 학습 가능한 파라미터 생성
* **`call()`**: 순전파 연산 정의

---

## 4. Keras Model 구성 방법

Keras에서는 크게 세 가지 방법으로 모델을 구성할 수 있습니다.

| 방법                    | 특징                      | 주요 용도           |
| --------------------- | ----------------------- | --------------- |
| **Sequential API**    | Layer를 순차적으로 연결         | 단순한 신경망         |
| **Functional API**    | 계산 그래프 형태로 모델 구성        | 다중 입출력 및 복잡한 모델 |
| **Model Subclassing** | Python 코드로 모델 동작을 직접 정의 | 사용자 정의 모델       |

### 4.1 Sequential API

```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

장점:

* 구조가 단순하고 직관적
* 빠르게 모델 구성 가능

한계:

* 순차적인 Layer 구조에 적합
* 복잡한 분기 구조에는 적합하지 않음

### 4.2 Functional API

```python
inputs = keras.Input(shape=(3,))
x = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

장점:

* 다중 입력 및 출력 지원
* Layer 공유 가능
* 비선형적인 계산 그래프 구성 가능
* 모델 구조 시각화가 용이

### 4.3 Model Subclassing

```python
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
```

장점:

* 높은 자유도
* 복잡한 순전파 로직 구현 가능

한계:

* Functional API보다 모델 구조가 명시적이지 않음
* 직렬화 및 시각화 시 추가 처리가 필요할 수 있음
* 구현해야 하는 부분이 많음

---

## 5. Model Compile 및 Training

### Compile

모델 학습 전 세 가지 주요 요소를 설정합니다.

| 구성 요소             | 역할           | 예시                                                                  |
| ----------------- | ------------ | ------------------------------------------------------------------- |
| **Loss Function** | 예측 오차 측정     | `BinaryCrossentropy`, `CategoricalCrossentropy`, `MeanSquaredError` |
| **Optimizer**     | 모델 파라미터 업데이트 | `SGD`, `RMSprop`, `Adam`, `Adagrad`                                 |
| **Metric**        | 모델 성능 평가     | `Accuracy`, `Precision`, `Recall`, `MAE`                            |

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Training

```python
history = model.fit(
    inputs,
    targets,
    epochs=5,
    batch_size=128,
    validation_data=(val_inputs, val_targets)
)
```

주요 학습 요소:

* **Epoch**: 전체 학습 데이터를 한 번 학습하는 과정
* **Batch Size**: 한 번의 파라미터 업데이트에 사용하는 데이터 수
* **Validation Data**: 학습 중 일반화 성능을 평가하기 위한 데이터

---

## 6. 주요 머신러닝 문제

### 이진 분류

예시: IMDB 영화 리뷰 감성 분류

* **출력 Layer**: `Dense(1, activation="sigmoid")`
* **Loss Function**: `binary_crossentropy`
* **목적**: 두 개 클래스 중 하나를 예측

### 다중 클래스 분류

예시: Reuters 뉴스 주제 분류

* **출력 Layer**: `Dense(num_classes, activation="softmax")`
* **Loss Function**: `categorical_crossentropy` 또는 `sparse_categorical_crossentropy`
* **목적**: 여러 클래스 중 하나를 예측

### 회귀

예시: 주택 가격 예측

* **출력 Layer**: 일반적으로 활성화 함수가 없는 하나의 출력 뉴런
* **Loss Function**: `mean_squared_error`
* **Metric**: `mean_absolute_error`
* **전처리**: 특성 정규화가 중요한 경우가 많음

---

## 7. 검증과 일반화

모델의 성능은 학습에 직접 사용하지 않은 데이터를 이용하여 평가해야 합니다.

### 검증 방법

* **Hold-out Validation**
* **K-Fold Cross-Validation**
* **Training / Validation / Test Split**

### 주요 문제

* **Overfitting**: 학습 데이터 성능은 높지만 검증 데이터 성능이 낮은 상태
* **Underfitting**: 학습 데이터와 검증 데이터 모두에서 성능이 낮은 상태

일반화 성능을 높이는 대표적인 방법:

* Regularization
* Dropout
* Early Stopping
* Data Augmentation
* 학습 데이터 증가

---

## 8. Callback

Callback은 모델 학습 과정에서 추가적인 작업을 수행하도록 하는 기능입니다.

### 주요 Callback

* **`ModelCheckpoint`**: 학습 중 모델 또는 가중치 저장
* **`EarlyStopping`**: 성능 개선이 멈추면 학습 조기 종료
* **`LearningRateScheduler`**: 학습률을 동적으로 변경
* **`ReduceLROnPlateau`**: 특정 지표가 개선되지 않을 때 학습률 감소

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3
    )
]
```

---

## 9. 사용자 정의 Metric

`keras.metrics.Metric`을 상속하여 사용자 정의 평가 지표를 만들 수 있습니다.

```python
class RootMeanSquaredError(keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        pass

    def reset_state(self):
        pass
```

주요 메서드:

* **`update_state()`**: Metric 내부 상태 업데이트
* **`result()`**: 최종 Metric 값 계산
* **`reset_state()`**: 저장된 상태 초기화

---

## 10. 다중 입력 및 다중 출력 모델

Functional API를 사용하면 서로 다른 여러 데이터를 입력받거나 여러 종류의 예측값을 출력하는 모델을 만들 수 있습니다.

예시:

* 텍스트 + 수치형 데이터
* 이미지 + 메타데이터
* 분류 + 회귀 출력

```python
model = keras.Model(
    inputs=[input_a, input_b],
    outputs=[output_a, output_b]
)
```

## 11. 합성곱 신경망 (Convolutional Neural Networks)

합성곱 신경망(CNN)은 이미지 처리와 컴퓨터 비전 분야에서 주로 사용되는 신경망 구조입니다.

### Conv2D

`Conv2D`는 이미지에 합성곱 필터를 적용하여 모서리, 질감, 형태와 같은 공간적 특징을 추출합니다.

```python
layers.Conv2D(
    32,
    kernel_size=(3, 3),
    activation="relu"
)
```


하나의 모델에서 여러 종류의 데이터를 함께 처리하거나 서로 연관된 여러 작업을 동시에 학습할 때 활용할 수 있습니다.

</details>
