<details>
<summary>ENG (English Version)</summary>

# Rock Paper Scissors CNN Classification

This project builds a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images into Rock, Paper, and Scissors.

A basic CNN was first trained as a baseline model. The model was then improved using **Data Augmentation, Dropout, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint**.

In addition to evaluating the model on the original Kaggle test dataset, I also tested it on **15 images captured in a real-world environment** to examine how well the model generalizes beyond the original dataset.

---

## 1. Project Overview

### Objective

The goal of this project is to classify input images into three classes:

* `paper`
* `rock`
* `scissors`

Rather than focusing only on training accuracy, this project explores the complete image classification workflow:

```text
Dataset
   ↓
Baseline CNN
   ↓
Model Evaluation
   ↓
Generalization Improvement
   ↓
Improved CNN
   ↓
Kaggle Test Evaluation
   ↓
Real-World Image Evaluation
```

The main objective was to compare the baseline and improved models and investigate whether strong performance on the original test dataset also transfers to images captured under different conditions.

### Development Environment

* Google Colab
* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* KaggleHub

---

## 2. Dataset

The Rock Paper Scissors Dataset from Kaggle was used.

```text
sanikamal/rock-paper-scissors-dataset
```

All images were resized to:

```text
150 × 150 × 3
```

Dataset structure:

```text
Training Dataset
    │
    ├── 80% Training
    │
    └── 20% Validation

Test Dataset
    └── Separate Test Images
```

Main configuration:

```python
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42
```

The datasets were created using Keras `image_dataset_from_directory()`.

---

## 3. Data Preprocessing

Image pixel values were normalized inside the model using:

```python
layers.Rescaling(1.0 / 255)
```

This converts pixel values from:

```text
0 ~ 255
   ↓
0 ~ 1
```

`tf.data.AUTOTUNE` and `prefetch()` were also used to improve the efficiency of the input pipeline during training.

---

## 4. Baseline CNN

A basic CNN without Data Augmentation or regularization was first trained as the baseline model.

### Architecture

```text
Input (150 × 150 × 3)
        ↓
Rescaling
        ↓
Conv2D (32)
        ↓
MaxPooling2D
        ↓
Conv2D (64)
        ↓
MaxPooling2D
        ↓
Conv2D (128)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dense (3, Softmax)
```

### Compilation

```python
baseline_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Because the class labels are represented as integer values rather than one-hot encoded vectors, `sparse_categorical_crossentropy` was used as the loss function.

---

## 5. Baseline Model Result

The Baseline CNN performed well on the training and validation datasets but showed considerably weaker performance on the separate test dataset.

| Model        | Test Loss | Test Accuracy |
| ------------ | --------: | ------------: |
| Baseline CNN |    1.7392 |    **78.49%** |

This indicated that the baseline model had limited generalization ability.

---

## 6. Model Improvement

Several techniques were applied to improve generalization performance.

### Data Augmentation

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
```

Data augmentation generates variations of the training images so that the model becomes less dependent on specific visual patterns.

### Dropout

```python
layers.Dropout(0.5)
```

Dropout randomly disables a portion of the neurons during training to reduce overfitting.

### EarlyStopping

```python
keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

Training stops when the validation loss no longer improves, and the weights from the best epoch are restored.

### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)
```

The learning rate is reduced when validation loss stops improving.

### ModelCheckpoint

`ModelCheckpoint` was used to save the model with the best validation loss during training.

---

## 7. Improved CNN

The CNN structure itself was kept mostly identical to the baseline model so that the effects of augmentation and regularization could be compared more clearly.

```text
Input
        ↓
Data Augmentation
        ↓
Rescaling
        ↓
Conv2D (32)
        ↓
MaxPooling2D
        ↓
Conv2D (64)
        ↓
MaxPooling2D
        ↓
Conv2D (128)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dropout (0.5)
        ↓
Dense (3, Softmax)
```

---

## 8. Model Performance Comparison

| Model        |  Test Loss | Test Accuracy |
| ------------ | ---------: | ------------: |
| Baseline CNN |     1.7392 |        78.49% |
| Improved CNN | **0.2013** |    **96.24%** |

The Improved CNN increased test accuracy from **78.49% to 96.24%**, while substantially reducing test loss.

This suggests that Data Augmentation, Dropout, and training callbacks improved the model's performance on the original test dataset.

---

## 9. Classification Report

The Improved CNN was evaluated on 372 Kaggle test images.

| Class             | Precision |   Recall | F1-score | Support |
| ----------------- | --------: | -------: | -------: | ------: |
| Paper             |      1.00 |     0.91 |     0.95 |     124 |
| Rock              |      1.00 |     0.98 |     0.99 |     124 |
| Scissors          |      0.90 |     1.00 |     0.95 |     124 |
| **Accuracy**      |           |          | **0.96** | **372** |
| **Macro Average** |  **0.97** | **0.96** | **0.96** | **372** |

### Result Analysis

#### Rock

Rock showed the strongest classification performance.

```text
Precision : 1.00
Recall    : 0.98
F1-score  : 0.99
```

#### Paper

Predictions classified as Paper were highly precise, although some actual Paper images were classified as other classes.

```text
Precision : 1.00
Recall    : 0.91
F1-score  : 0.95
```

#### Scissors

All actual Scissors images were detected, but some images from other classes were incorrectly classified as Scissors.

```text
Precision : 0.90
Recall    : 1.00
F1-score  : 0.95
```

---

## 10. Confusion Matrix

A confusion matrix was generated using Scikit-learn.

```python
cm = confusion_matrix(
    y_true,
    y_pred
)
```

This made it possible to analyze class-level prediction errors in addition to overall accuracy.

---

## 11. Prediction Visualization

Predictions from the test dataset were visualized with:

```text
Actual Class
Predicted Class
Prediction Confidence
```

This allowed both quantitative metrics and individual prediction results to be examined.

---

## 12. Real-World Generalization Test

To test the model outside the original Kaggle dataset, I captured **15 new images** under real-world conditions.

The dataset consisted of:

* 5 Rock images
* 5 Paper images
* 5 Scissors images

### Results

| Class       | Correct |  Total |   Accuracy |
| ----------- | ------: | -----: | ---------: |
| Rock        |       2 |      5 |        40% |
| Paper       |       1 |      5 |        20% |
| Scissors    |       4 |      5 |        80% |
| **Overall** |   **7** | **15** | **46.67%** |

The model achieved **96.24% accuracy on the original Kaggle test dataset**, but only **46.67% accuracy on personally captured images**.

```text
Kaggle Test Accuracy
96.24%

        ↓

Real-World Image Accuracy
46.67%
```

This revealed a significant generalization gap between the original dataset and real-world images.

Some incorrect predictions were also made with high confidence, indicating that the model could be highly confident even when its prediction was incorrect.

Possible causes include:

* Different backgrounds
* Different lighting conditions
* Camera angle changes
* Hand orientation
* Hand size and position within the image
* Differences between the training image distribution and real-world images

This suggests that the model may have learned not only the hand shapes themselves, but also visual patterns associated with the original dataset.

---

## 13. Model Saving

The final Improved CNN was saved in Keras format.

```python
improved_model.save(
    "/content/rps_cnn_model.keras"
)
```

The saved model can be reused for future predictions without retraining.

---

## 14. What I Learned

Through this project, I practiced:

* Building CNN models with TensorFlow and Keras
* Image feature extraction using `Conv2D`
* Feature map reduction using `MaxPooling2D`
* Multi-class image classification
* Using a `Softmax` output layer
* Using `sparse_categorical_crossentropy`
* Training / Validation / Test dataset separation
* Data Augmentation
* Dropout regularization
* EarlyStopping
* ReduceLROnPlateau
* ModelCheckpoint
* Training and validation curve analysis
* Precision, Recall, and F1-score evaluation
* Confusion Matrix analysis
* Saving and reusing trained models
* Predicting multiple custom images
* Evaluating model performance on out-of-distribution data
* Identifying generalization problems and domain shift

---

## 15. Conclusion

The Baseline CNN achieved **78.49% Test Accuracy** on the original test dataset.

After applying Data Augmentation, Dropout, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint, the Improved CNN achieved **96.24% Test Accuracy** with a Test Loss of **0.2013**.

However, when the model was evaluated on 15 personally captured images, accuracy decreased to **46.67%**.

This experiment showed that strong performance on an internal test dataset does not necessarily guarantee strong performance in a different real-world environment.

The project therefore progressed beyond simply improving CNN accuracy and demonstrated the importance of:

* Evaluating models on unseen environments
* Detecting domain shift
* Examining incorrect high-confidence predictions
* Considering dataset diversity when developing image classification systems

Through this project, I practiced the complete image classification workflow from **model construction and training to regularization, evaluation, real-world testing, and generalization analysis**.

---

## 16. Try It Yourself

You can test the trained model with your own Rock, Paper, or Scissors images in Google Colab.

1. Open the project notebook in Google Colab.
2. Run the notebook cells in order.
3. Go to the final cell: `Predict Multiple Custom Images`.
4. Run the cell and upload one or more images.
5. The model will display the predicted class, confidence score, and probability for each class.

Example:

```text id="z67ij4"
Upload Image
     ↓
Resize to 150 × 150
     ↓
CNN Model
     ↓
Prediction
     ↓
Rock / Paper / Scissors
```

The final cell supports multiple image uploads at once.

```python id="y67j57"
uploaded = files.upload()
```

After uploading your images, the model will display:

```text id="87kejx"
Prediction
Confidence
Paper Probability
Rock Probability
Scissors Probability
```

> The model may perform differently on real-world images because backgrounds, lighting, camera angles, and hand positions can differ from the original training dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Rx042ATmA4qHK3Dh_s7ywVLUEqZcqPPj?hl=ko)


</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 가위바위보 CNN 이미지 분류

TensorFlow와 Keras를 사용하여 Rock, Paper, Scissors 이미지를 분류하는 CNN(Convolutional Neural Network) 모델을 구축한 프로젝트입니다.

먼저 기본 CNN을 Baseline 모델로 학습한 뒤, **Data Augmentation, Dropout, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint**를 적용하여 모델을 개선했습니다.

또한 Kaggle의 기존 Test Dataset뿐만 아니라 **직접 촬영한 이미지 15장**을 이용하여 실제 환경에서도 모델이 일반화되는지 추가로 검증했습니다.

---

## 1. 프로젝트 개요

### 목표

입력 이미지를 다음 3개의 클래스로 분류하는 모델을 구축하는 것이 목표입니다.

* `paper`
* `rock`
* `scissors`

단순히 학습 정확도를 높이는 것보다 다음과 같은 전체 이미지 분류 과정을 실습하는 데 중점을 두었습니다.

```text
데이터셋
   ↓
기본 CNN
   ↓
모델 평가
   ↓
일반화 성능 개선
   ↓
개선 CNN
   ↓
Kaggle 테스트
   ↓
실제 촬영 이미지 테스트
```

특히 기존 Test Dataset에서 높은 성능을 기록한 모델이 실제 환경에서 촬영한 새로운 이미지에서도 동일하게 동작하는지를 확인했습니다.

### 개발 환경

* Google Colab
* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* KaggleHub

---

## 2. 데이터셋

Kaggle의 Rock Paper Scissors Dataset을 사용했습니다.

```text
sanikamal/rock-paper-scissors-dataset
```

모든 이미지는 다음 크기로 조정했습니다.

```text
150 × 150 × 3
```

데이터 구성:

```text
학습용 데이터
    │
    ├── 80% Training
    │
    └── 20% Validation

테스트 데이터
    └── 별도의 Test Images
```

주요 설정:

```python
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42
```

Keras의 `image_dataset_from_directory()`를 사용하여 이미지 폴더에서 데이터셋을 생성했습니다.

---

## 3. 데이터 전처리

입력 이미지의 픽셀 값은 모델 내부에서 다음과 같이 정규화했습니다.

```python
layers.Rescaling(1.0 / 255)
```

이를 통해 픽셀 범위를 다음과 같이 변환했습니다.

```text
0 ~ 255
   ↓
0 ~ 1
```

또한 `tf.data.AUTOTUNE`과 `prefetch()`를 사용하여 학습 과정에서 데이터가 보다 효율적으로 공급되도록 구성했습니다.

---

## 4. 기본 CNN 모델

먼저 Data Augmentation이나 Regularization을 적용하지 않은 기본 CNN을 Baseline 모델로 구성했습니다.

### 모델 구조

```text
Input (150 × 150 × 3)
        ↓
Rescaling
        ↓
Conv2D (32)
        ↓
MaxPooling2D
        ↓
Conv2D (64)
        ↓
MaxPooling2D
        ↓
Conv2D (128)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dense (3, Softmax)
```

### 모델 컴파일

```python
baseline_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

클래스 라벨이 One-Hot Encoding이 아닌 정수 형태이므로 `sparse_categorical_crossentropy`를 Loss Function으로 사용했습니다.

---

## 5. 기본 모델 결과

Baseline CNN은 Training 및 Validation Dataset에서는 높은 성능을 보였지만 별도의 Test Dataset에서는 성능이 감소했습니다.

| 모델           | Test Loss | Test Accuracy |
| ------------ | --------: | ------------: |
| Baseline CNN |    1.7392 |    **78.49%** |

이를 통해 Baseline 모델이 학습 데이터와 유사한 환경에는 잘 맞지만 새로운 데이터에 대한 **일반화 성능이 부족한 문제**를 확인했습니다.

---

## 6. 모델 개선

일반화 성능을 높이기 위해 여러 방법을 적용했습니다.

### 데이터 증강

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
```

학습 이미지에 다양한 변형을 적용하여 모델이 특정 이미지 패턴에 지나치게 의존하는 것을 줄였습니다.

### 드롭아웃

```python
layers.Dropout(0.5)
```

학습 과정에서 일부 뉴런을 무작위로 비활성화하여 Overfitting을 줄였습니다.

### 조기 종료

```python
keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

Validation Loss가 더 이상 개선되지 않을 경우 학습을 중단하고 가장 좋은 시점의 Weight를 복원했습니다.

### 학습률 감소

```python
keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)
```

Validation Loss가 개선되지 않을 경우 Learning Rate를 감소시켜 보다 세밀한 학습이 가능하도록 했습니다.

### 최적 모델 저장

`ModelCheckpoint`를 사용하여 Validation Loss가 가장 낮은 모델을 저장하도록 구성했습니다.

---

## 7. 개선 CNN 모델

Baseline CNN과의 비교를 명확하게 하기 위해 CNN의 기본적인 구조는 최대한 동일하게 유지했습니다.

```text
Input
        ↓
Data Augmentation
        ↓
Rescaling
        ↓
Conv2D (32)
        ↓
MaxPooling2D
        ↓
Conv2D (64)
        ↓
MaxPooling2D
        ↓
Conv2D (128)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dropout (0.5)
        ↓
Dense (3, Softmax)
```

이를 통해 CNN 구조 자체의 변화보다는 **Data Augmentation과 Regularization이 일반화 성능에 미치는 영향**을 비교했습니다.

---

## 8. 모델 성능 비교

| 모델           |  Test Loss | Test Accuracy |
| ------------ | ---------: | ------------: |
| Baseline CNN |     1.7392 |        78.49% |
| Improved CNN | **0.2013** |    **96.24%** |

Improved CNN은 Baseline CNN보다 Test Loss가 크게 감소했고, Test Accuracy는 **78.49%에서 96.24%까지 증가**했습니다.

기존 Kaggle Test Dataset에서는 Data Augmentation, Dropout 및 Callback을 이용한 학습 전략이 일반화 성능 개선에 효과적이었습니다.

---

## 9. 분류 성능 평가

Improved CNN을 Kaggle Test Image 372장에 대해 평가했습니다.

| 클래스               | Precision |   Recall | F1-score | Support |
| ----------------- | --------: | -------: | -------: | ------: |
| Paper             |      1.00 |     0.91 |     0.95 |     124 |
| Rock              |      1.00 |     0.98 |     0.99 |     124 |
| Scissors          |      0.90 |     1.00 |     0.95 |     124 |
| **Accuracy**      |           |          | **0.96** | **372** |
| **Macro Average** |  **0.97** | **0.96** | **0.96** | **372** |

### 결과 분석

#### 바위

세 클래스 중 가장 안정적인 분류 성능을 나타냈습니다.

```text
Precision : 1.00
Recall    : 0.98
F1-score  : 0.99
```

#### 보

Paper로 분류한 이미지의 정확도는 높았지만 일부 실제 Paper 이미지를 다른 클래스로 분류하는 경우가 있었습니다.

```text
Precision : 1.00
Recall    : 0.91
F1-score  : 0.95
```

#### 가위

실제 Scissors 이미지는 모두 탐지했지만 일부 다른 클래스 이미지를 Scissors로 잘못 분류하는 경우가 있었습니다.

```text
Precision : 0.90
Recall    : 1.00
F1-score  : 0.95
```

---

## 10. 혼동 행렬

Scikit-learn을 사용하여 Confusion Matrix를 생성했습니다.

```python
cm = confusion_matrix(
    y_true,
    y_pred
)
```

이를 통해 전체 Accuracy뿐만 아니라 클래스별 오분류 패턴도 확인했습니다.

---

## 11. 예측 결과 시각화

Test Dataset의 각 이미지에 대해 다음 정보를 함께 시각화했습니다.

```text
실제 클래스
예측 클래스
예측 신뢰도
```

이를 통해 정량적인 평가 결과뿐만 아니라 개별 이미지에 대한 모델의 실제 예측 결과도 확인했습니다.

---

## 12. 실제 촬영 이미지 일반화 테스트

Kaggle Dataset 이외의 환경에서도 모델이 제대로 작동하는지 확인하기 위해 스마트폰으로 직접 **15장의 이미지**를 촬영했습니다.

테스트 이미지는 다음과 같이 구성했습니다.

* Rock 5장
* Paper 5장
* Scissors 5장

### 테스트 결과

| 클래스      |    정답 |     전체 |        정확도 |
| -------- | ----: | -----: | ---------: |
| Rock     |     2 |      5 |        40% |
| Paper    |     1 |      5 |        20% |
| Scissors |     4 |      5 |        80% |
| **전체**   | **7** | **15** | **46.67%** |

기존 Kaggle Test Dataset에서는 **96.24%의 Accuracy**를 기록했지만 직접 촬영한 이미지에서는 **46.67%**까지 감소했습니다.

```text
Kaggle Test Accuracy
96.24%

        ↓

실제 촬영 이미지 Accuracy
46.67%
```

이는 기존 데이터셋과 실제 촬영 환경 사이에 상당한 **일반화 성능 차이**가 존재한다는 것을 보여줍니다.

또한 일부 잘못된 예측에서도 높은 Confidence를 나타냈습니다.

즉 모델이 잘못된 판단을 하면서도 높은 확률로 특정 클래스를 예측하는 경우가 존재했습니다.

이러한 결과의 원인으로 다음과 같은 차이를 고려할 수 있습니다.

* 배경 차이
* 조명 변화
* 카메라 각도
* 손의 방향
* 이미지 내 손의 크기와 위치
* 학습 데이터와 실제 이미지 사이의 데이터 분포 차이

따라서 모델이 손 모양 자체뿐만 아니라 기존 데이터셋의 배경이나 촬영 환경과 같은 시각적 특징에도 일정 부분 의존했을 가능성이 있습니다.

---

## 13. 모델 저장

최종 Improved CNN은 Keras 모델 형식으로 저장했습니다.

```python
improved_model.save(
    "/content/rps_cnn_model.keras"
)
```

저장된 모델은 다시 학습할 필요 없이 이후 Prediction에 재사용할 수 있습니다.

---

## 14. 학습한 내용

이 프로젝트를 통해 다음 내용을 실제 이미지 분류 문제에 적용했습니다.

* TensorFlow / Keras 기반 CNN 모델 구축
* `Conv2D`를 이용한 이미지 특징 추출
* `MaxPooling2D`를 이용한 Feature Map 축소
* 다중 클래스 이미지 분류
* `Softmax` 출력층 구성
* `sparse_categorical_crossentropy` Loss Function 사용
* Training / Validation / Test Dataset 분리
* Data Augmentation
* Dropout을 이용한 Regularization
* EarlyStopping
* ReduceLROnPlateau
* ModelCheckpoint
* Training / Validation Curve 분석
* Precision / Recall / F1-score 평가
* Confusion Matrix 분석
* 학습 모델 저장 및 재사용
* 여러 개의 새로운 이미지 일괄 예측
* 기존 데이터 분포와 다른 이미지에 대한 모델 평가
* 일반화 문제와 Domain Shift 확인

---

## 15. 결론

Baseline CNN은 기존 Test Dataset에서 **78.49%의 Accuracy**를 기록했습니다.

이후 Data Augmentation, Dropout, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint를 적용하여 개선한 CNN은 **96.24%의 Test Accuracy와 0.2013의 Test Loss**를 기록했습니다.

하지만 모델을 직접 촬영한 이미지 15장에 적용한 결과 정확도는 **46.67%**까지 감소했습니다.

이를 통해 기존 Test Dataset에서 높은 정확도를 기록하는 것만으로는 실제 환경에서도 동일한 성능을 보장할 수 없다는 것을 확인했습니다.

특히 이번 프로젝트에서는 단순히 CNN의 정확도를 높이는 것에서 끝나지 않고 다음 문제까지 확인할 수 있었습니다.

* 새로운 환경에서의 모델 성능 저하
* 데이터 분포 차이에 따른 Domain Shift
* 높은 Confidence를 가진 오분류
* 실제 환경을 고려한 데이터 다양성의 중요성

따라서 이 프로젝트를 통해 **CNN 모델 구축 → 학습 → 일반화 성능 개선 → 평가 → 실제 이미지 테스트 → 일반화 문제 분석**까지 이어지는 전체 이미지 분류 과정을 실습했습니다.

---

## 16. 직접 테스트해보기

Google Colab에서 직접 촬영한 가위, 바위, 보 이미지를 업로드하여 모델을 테스트할 수 있습니다.

1. 프로젝트 Notebook을 Google Colab에서 엽니다.
2. 위에서부터 셀을 순서대로 실행합니다.
3. 마지막의 `Predict Multiple Custom Images` 셀로 이동합니다.
4. 해당 셀을 실행한 뒤 하나 이상의 이미지를 업로드합니다.
5. 모델이 각 이미지의 예측 결과와 Confidence, 클래스별 확률을 출력합니다.

동작 과정:

```text id="18vhep"
이미지 업로드
     ↓
150 × 150 크기로 변환
     ↓
CNN 모델
     ↓
예측
     ↓
Rock / Paper / Scissors
```

마지막 셀에서는 여러 장의 이미지를 한 번에 업로드할 수 있습니다.

```python id="mj5gjh"
uploaded = files.upload()
```

이미지를 업로드하면 다음 정보를 확인할 수 있습니다.

```text id="j5x7l6"
예측 클래스
예측 신뢰도
Paper 확률
Rock 확률
Scissors 확률
```

> 실제 촬영 이미지는 학습 데이터와 배경, 조명, 촬영 각도, 손의 위치 등이 다르기 때문에 기존 Test Dataset보다 성능이 낮게 나타날 수 있습니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Rx042ATmA4qHK3Dh_s7ywVLUEqZcqPPj?hl=ko)

</details>
