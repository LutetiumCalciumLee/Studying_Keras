<details>
<summary>ENG (English Version)</summary>

# Keras Theory

## 1. Machine Learning Foundations and High-Level Deep Learning

### TensorFlow (Low-Level Tensor Operations)
- **Tensor**: The basic data structure that includes both constants and variables, and is fundamentally constant in nature.
- **Tensor Operations**: Supports mathematical operations such as addition, `relu`, `matmul`, etc.
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
  - For constant tensors, manual tracking is needed with `tape.watch()`.
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
  - Cannot output layer connection structure with `summary()`
  - Cannot visualize structure with `plot_model()`
  - Cannot use layers externally

## 5. Model Training Process

### 5.1 Compile
Three key elements to decide for training:

1.  **Loss Function**: The value to minimize during training
    -   `CategoricalCrossentropy`, `SparseCategoricalCrossentropy`, `BinaryCrossentropy`
2.  **Optimizer**: Determines how the network is updated
    -   `SGD`, `RMSprop`, `Adam`, `Adagrad`
3.  **Metrics**: Success indicators to monitor during training and validation
    -   `CategoricalAccuracy`, `SparseCategoricalAccuracy`, `BinaryAccuracy`

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
- **Loss Function**: `binary_crossentropy`

### 6.2 Multi-Class Classification (Reuters News Classification)
- **Data**: News articles from 46 topics
- **Preprocessing**: One-hot encoding
- **Model Structure**: Dense-Dense-Dense with softmax output
- **Loss Function**: `categorical_crossentropy`

### 6.3 Regression (Boston Housing Price Prediction)
- **Data**: 404 training samples, 102 test samples
- **Preprocessing**: Data normalization required
- **Validation Method**: k-fold cross-validation
- **Loss Function**: `mean_squared_error`

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

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 케라스 이론

## 1. 머신러닝 기초와 고수준 딥러닝

### 텐서플로우 (저수준 텐서 연산)
- **텐서**: 상수와 변수를 포함하는 기본 데이터 구조이며, 근본적으로는 불변성을 가짐.
- **텐서 연산**: 덧셈, `relu`, `matmul` 등 수학적 연산을 지원.
- **역전파**: 신경망 훈련 중 그래디언트 계산을 위한 핵심 방법.

### 케라스 (고수준 딥러닝 API)
- **모델 레이어**: 딥러닝 모델의 기본 구성 요소.
- **손실 함수**: 모델 성능을 측정하는 기준.
- **옵티마이저**: 가중치 업데이트 방식을 결정.
- **메트릭**: 모델 성능을 평가하기 위한 지표.
- **훈련 루프**: 미니배치 확률적 경사 하강법을 수행.

## 2. 텐서플로우에서의 텐서 연산

### 상수 텐서
```python
# 주요 텐서 생성 함수
tf.ones(shape=(2, 1))      # 1로 채워진 텐서
tf.zeros(shape=(2, 1))     # 0으로 채워진 텐서
tf.random.normal()         # 정규분포에서 무작위 숫자 추출
tf.random.uniform()        # 균등분포에서 무작위 숫자 추출
```

### 변수 텐서
- **`tf.Variable`**: 변경 가능한 상태를 관리하는 클래스.
- **`assign()`**: 변수에 값을 할당.
- **`assign_add()`**: 변수 값에 더하기.
- **`assign_sub()`**: 변수 값에서 빼기.

### 그래디언트 계산
- **`tf.GradientTape()`**: 그래디언트 계산을 위한 일반 도구.
  - 훈련 가능한 변수 텐서만 자동으로 추적.
  - 상수 텐서는 `tape.watch()`로 수동 추적 필요.
  - 고차 도함수 지원.

## 3. 케라스 레이어 개념

### 레이어 속성
- **정의**: 하나 이상의 텐서를 입력으로 받아 하나 이상의 텐서를 출력하는 데이터 처리 모듈.
- **가중치**: 확률적 경사 하강법을 통해 학습되는 하나 이상의 텐서.
- **텐서 형식**: 각 레이어는 다른 텐서 형태를 지원:
  - 2차원 텐서 (샘플, 특성): 완전 연결 레이어 (`Dense`)
  - 3차원 텐서 (샘플, 타임스텝, 특성): 순환 레이어, 1D 합성곱 레이어

### 케라스 레이어 클래스 예시
```python
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        # 레이어 매개변수 정의
    
    def build(self, input_shape):
        # 가중치 생성 (첫 호출 시 자동 실행)
    
    def call(self, inputs):
        # 순전파 계산 정의
```

## 4. 케라스 모델 구성 방법

### 4.1 순차 모델
- **특징**: 가장 간단한 방식으로, 레이어를 순차적으로 쌓음.
- **한계**: 단일 입력, 단일 출력, 그리고 순차적인 구조만 지원.
- **사용 사례**: 기본적인 신경망 구조.

```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

### 4.2 함수형 API
- **특징**: 그래프와 같은 모델 구조를 다룰 수 있음.
- **장점**: 사용성과 유연성 사이의 좋은 균형.
- **사용 사례**: 다중 입력/출력, 비선형 구조.

```python
inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)
```

### 4.3 모델 서브클래싱
- **특징**: 처음부터 모델을 만드는 저수준 방식.
- **장점**: 모든 세부 사항을 완벽하게 제어.
- **단점**: 
  - `summary()`로 레이어 연결 구조 출력 불가
  - `plot_model()`로 구조 시각화 불가
  - 레이어를 외부에서 사용 불가

## 5. 모델 훈련 과정

### 5.1 컴파일
훈련을 위해 결정해야 할 세 가지 핵심 요소:

1.  **손실 함수**: 훈련 중에 최소화할 값.
    -   `CategoricalCrossentropy`, `SparseCategoricalCrossentropy`, `BinaryCrossentropy`
2.  **옵티마이저**: 신경망 업데이트 방식 결정.
    -   `SGD`, `RMSprop`, `Adam`, `Adagrad`
3.  **메트릭**: 훈련 및 검증 중 모니터링할 성공 지표.
    -   `CategoricalAccuracy`, `SparseCategoricalAccuracy`, `BinaryAccuracy`

### 5.2 훈련 (`Fit`)
```python
model.fit(
    inputs,
    targets,
    epochs=5,           # 훈련 에포크 수
    batch_size=128,     # 배치 크기
    validation_data=(val_inputs, val_targets)  # 검증 데이터
)
```

## 6. 주요 적용 사례

### 6.1 이진 분류 (IMDB 영화 리뷰)
- **데이터**: 5만 개의 영화 리뷰 (긍정/부정)
- **전처리**: 멀티-핫 인코딩으로 벡터화
- **모델 구조**: `Dense-Dense-Dense` 구조에 `sigmoid` 출력
- **손실 함수**: `binary_crossentropy`

### 6.2 다중 클래스 분류 (로이터 뉴스 분류)
- **데이터**: 46개 주제의 뉴스 기사
- **전처리**: 원-핫 인코딩
- **모델 구조**: `Dense-Dense-Dense` 구조에 `softmax` 출력
- **손실 함수**: `categorical_crossentropy`

### 6.3 회귀 (보스턴 주택 가격 예측)
- **데이터**: 404개의 훈련 샘플, 102개의 테스트 샘플
- **전처리**: 데이터 정규화 필요
- **검증 방법**: K-겹 교차 검증
- **손실 함수**: `mean_squared_error`

## 7. 고급 기능

### 7.1 콜백
훈련 중 모델 상태를 모니터링하고 제어하는 도구:
- **`ModelCheckpoint`**: 모델 가중치 저장
- **`EarlyStopping`**: 조기 종료
- **`LearningRateScheduler`**: 동적 학습률 조정

### 7.2 사용자 정의 메트릭
```python
class RootMeanSquaredError(keras.metrics.Metric):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # 상태 업데이트 로직
    
    def result(self):
        # 최종 결과 계산
    
    def reset_state(self):
        # 상태 초기화
```

### 7.3 다중 입력/출력 모델
- **입력**: 다양한 유형의 데이터 (텍스트, 범주형 등)
- **출력**: 다양한 유형의 예측 (우선순위, 분류 등)
- **구현**: 함수형 API 사용

</details>
