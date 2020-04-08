---
layout: post
title: Image Classification 실습해보기 (AlexNet으로 개vs고양이 분류)
category: deep learning
tags: [classification, alexnet, tensorflow]
comments: true

---

## 이미지 인식 문제를 위한 딥러닝의 기본 요소
- 데이터셋
- 성능 평가
- (딥)러닝 모델
- (딥)러닝 알고리즘

그러면 지금부터, 위에서 언급한 딥러닝의 4가지 기본 요소를 기준으로 삼아, ‘개vs고양이 분류’ 문제 해결을 위해 직접 제작한 AlexNet 구현체를 소개해 드리도록 하겠습니다.
<br/>

## (1) 데이터셋: Asirra Dogs vs. Cats dataset

개vs고양이 분류 문제를 위해 사용한 데이터셋의 원본은 The Asirra dataset이며, 본 글에서 실제 사용한 데이터셋은 데이터 사이언스 관련 유명 웹사이트인 Kaggle에서 제공하는 competitions 항목 중 Dogs vs. Cats로부터 가져온 것입니다.

- 원본 데이터셋 구성
traning dataset: 25,000장 -> 학습 데이터셋에 대해서만 라벨링이 되어 제공됨
test dataset: 12,500장

이번 실습에서는 개, 고양이 분류 문제 세팅을 위해서 원본 학습 데이터셋 중 랜덤하게 절반크기만큼 샘플링하여 12,500을 학습 데이터셋으로 나머지의 데이터를 테스트 데이터셋으로 재정의하였습니다.

이미지 크기는 가로 42~1050픽셀, 세로 32~768픽셀 사이에서 가변적입니다. 개와 고양이 분류 문제용 데이터셋이므로, 클래스는 0(고양이), 1(개)의 이진 클래스로 구성되어 있습니다.

### datasets.asirra 모듈
이 모듈은 데이터셋 요소에 해당하는 모든 함수들과 클래스를 담고 있습니다. 이들 중에서, 1. 디스크로부터 데이터셋을 메모리에 로드하고, 2. 학습 및 예측 과정에서 이들을 미니배치(minibatch) 단위로 추출하는 부분을 중심으로 살펴보도록 하겠습니다.

#### read_asirra_subset 함수
```python
def read_asirra_subset(subset_dir, one_hot=True, sample_size=None)
"""
1. 디스크로부터 데이터셋을 로드
2. AlexNet을 학습하기 위한 형태로 전처리 수행
:param subset_dir: str, 원본 데이터셋이 저장된 디렉터리 경로.
:param one_hot: bool, one-hot 인코딩 형태의 레이블을 반환할 것인지 여부.
:param sample_size: int, 전체 데이터셋을 모두 사용하지 않는 경우, 사용하고자 하는 샘플 이미지 개수.
:return: X_set: np.ndarray, shape: (N, H, W, C).
         y_set: np.ndarray, shape: (N, num_channels) or (N,).
"""
# 학습 + 검증 데이터 셋을 읽어들임
filename_list = os.listdir(subset_dir)
set_size = len(filename_list)

if sample_size is not None and sample_size < set_size:
    # sample_size가 명시된 경우, 원본 중 일부를 랜덤하게 샘플링함
    filename_list = np.random.choice(filename_list, size=sample_size,replace=False)
    set_size = sample_size
else:
    # 단순히 filename list의 순서를 랜덤하게 섞음
    np.random.shuffle(filename_list)

# 데이터 array들을 메모리 공간에 미리 할당함
X_set = np.empty((set_size, 256, 256, 3), dtype=np.float32)    # (N, H, W, 3)
y_set = np.empty((set_size), dtype=np.uint8)                   # (N,)
for i, filename in enumerate(filename_list):
    if i % 1000 == 0:
        print('Reading subset data: {}/{}...'.format(i,set_size), end='\r')
    label = filename.split('.')[0]
     if label == 'cat':
            y = 0
        else:  # label == 'dog'
            y = 1
        file_path = os.path.join(subset_dir, filename)
        img = imread(file_path)    # shape: (H, W, 3), range: [0, 255]
        img = resize(img, (256, 256), mode='constant').astype(np.float32)    # (256, 256, 3), [0.0, 1.0]
        X_set[i] = img
        y_set[i] = y

    if one_hot:
        # 모든 레이블들을 one-hot 인코딩 벡터들로 변환함, shape: (N, num_classes)
        y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
        y_set_oh[np.arange(set_size), y_set] = 1
        y_set = y_set_oh
    print('\nDone')

    return X_set, y_set
```

#### DataSet 클래스
```python
class DataSet(object):
    def __init__(self, images, labels=None):
        """
        새로운 DataSet 객체를 생성함.
        :param images: np.ndarray, shape: (N, H, W, C).
        :param labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if labels is not None:
            assert images.shape[0] == labels.shape[0], (
                'Number of examples mismatch, between images and labels.'
            )
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels    # NOTE: 만약 입력 인자로 주어지지 않았다면, None으로 남길 수 있음.
        self._indices = np.arange(self._num_examples, dtype=np.uint)    # image/label 인덱스 생성(추후 랜덤하게 섞일 수 있음)
        self._reset()

    def _reset(self):
        """일부 변수를 재설정함."""
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True, fake_data=False):
        """
        `batch_size` 개수만큼의 이미지들을 현재 데이터셋으로부터 추출하여 미니배치 형태로 반환함.
        :param batch_size: int, 미니배치 크기.
        :param shuffle: bool, 미니배치 추출에 앞서, 현재 데이터셋 내 이미지들의 순서를 랜덤하게 섞을 것인지 여부.
        :param augment: bool, 미니배치를 추출할 때, 데이터 증강을 수행할 것인지 여부.
        :param is_train: bool, 미니배치 추출을 위한 현재 상황(학습/예측).
        :param fake_data: bool, (디버깅 목적으로) 가짜 이미지 데이터를 생성할 것인지 여부.
        :return: batch_images: np.ndarray, shape: (N, h, w, C) or (N, 10, h, w, C).
                 batch_labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if fake_data:
            fake_batch_images = np.random.random(size=(batch_size, 227, 227, 3))
            fake_batch_labels = np.zeros((batch_size, 2), dtype=np.uint8)
            fake_batch_labels[np.arange(batch_size), np.random.randint(2, size=batch_size)] = 1
            return fake_batch_images, fake_batch_labels

        start_index = self._index_in_epoch
    
        # 맨 첫 번째 epoch에서는 전체 데이터셋을 랜덤하게 섞음 ---
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # 현재의 인덱스가 전체 이미지 수를 넘어간 경우, 다음 epoch을 진행함
        if start_index + batch_size > self._num_examples:
            # 완료된 epochs 수를 1 증가
            self._epochs_completed += 1
            # 새로운 epoch에서, 남은 이미지들을 가져옴
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # 하나의 epoch이 끝나면, 전체 데이터셋을 섞음
            if shuffle:
                np.random.shuffle(self._indices)

            # 다음 epoch 시작
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self.images[indices_rest_part]
            images_new_part = self.images[indices_new_part]
            batch_images = np.concatenate((images_rest_part, images_new_part), axis=0)
            if self.labels is not None:
                labels_rest_part = self.labels[indices_rest_part]
                labels_new_part = self.labels[indices_new_part]
                batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self.images[indices]
            if self.labels is not None:
                batch_labels = self.labels[indices]
            else:
                batch_labels = None

        if augment and is_train:
            # 학습 상황에서의 데이터 증강을 수행함 
            batch_images = random_crop_reflect(batch_images, 227)
        elif augment and not is_train:
            # 예측 상황에서의 데이터 증강을 수행함
            batch_images = corner_center_crop_reflect(batch_images, 227)
        else:
            # 데이터 증강을 수행하지 않고, 단순히 이미지 중심 위치에서만 추출된 패치를 사용함
            batch_images = center_crop(batch_images, 227)

        return batch_images, batch_labels
```
데이터셋 요소를 클래스화한 것이 DataSet 클래스입니다. 여기에는 기본적으로 이미지들과 이에 해당하는 레이블들이 np.ndarray 타입의 멤버로 포함되어 있습니다. 핵심이 되는 부분은 next_batch 함수인데, 이는 주어진 batch_size 크기의 미니배치(이미지, 레이블)를 현재 데이터셋으로부터 추출하여 반환합니다.

원 AlexNet 논문에서는 학습 단계와 테스트 단계에서의 데이터 증강(data augmentation) 방법을 아래와 같이 서로 다르게 채택하고 있습니다.
- 학습 단계: 원본 256×256 크기의 이미지로부터 227×227 크기의 패치(patch)를 랜덤한 위치에서 추출하고, 50% 확률로 해당 패치에 대한 수평 방향으로의 대칭 변환(horizontal reflection)을 수행하여, 이미지 하나 당 하나의 패치를 반환함
- 테스트 단계: 원본 256×256 크기 이미지에서의 좌측 상단, 우측 상단, 좌측 하단, 우측 하단, 중심 위치 각각으로부터 총 5개의 227×227 패치를 추출하고, 이들 각각에 대해 수평 방향 대칭 변환을 수행하여 얻은 5개의 패치를 추가하여, 이미지 하나 당 총 10개의 패치를 반환함

next_batch 함수에서는 데이터 증강을 수행하도록 설정되어 있는 경우에 한해(augment == True), **현재 학습 단계인지(is_train == True) 테스트 단계인지(is_train == False)에 따라** 위와 같이 서로 다른 데이터 증강 방법을 적용하고, 이를 통해 얻어진 패치 단위의 이미지들을 반환하도록 하였습니다.

#### 원 논문과의 차이점
본래 AlexNet 논문에서는 추출되는 패치의 크기가 224×224라고 명시되어 있으나, 본 구현체에서는 227×227로 하였습니다. 실제로 온라인 상의 많은 AlexNet 구현체에서 227×227 크기를 채택하고 있으며, 이렇게 해야만 올바른 형태로 구현이 가능합니다.

또, AlexNet 논문에서는 여기에 PCA에 기반한 색상 증강(color augmentation)을 추가로 수행하였는데, 본 구현체에서는 구현의 단순화를 위해 이를 반영하지 않았습니다.

<br/>

## (2) 성능 평가: 정확도
개vs고양이 분류 문제의 성능 평가 척도로는, 가장 단순한 척도인 정확도(accuracy)를 사용합니다. 단일 사물 분류 문제의 경우 주어진 이미지를 하나의 클래스로 분류하기만 하면 되기 때문에, 정확도가 가장 직관적인 척도라고 할 수 있습니다. 이는, 테스트를 위해 주어진 전체 이미지 수 대비, 분류 모델이 올바르게 분류한 이미지 수로 정의됩니다.

$$
\begin{equation}
\text{정확도} = \frac{\text{올바르게 분류한 이미지 수}} {\text{전체 이미지 수}}
\end{equation}
$$

### learning.evaluators 모듈
이 모듈은, 현재까지 학습된 모델의 성능 평가를 위한 ‘evaluator(성능 평가를 수행하는 개체)’의 클래스를 담고 있습니다.

#### Evaluator 클래스
```python
class Evaluator(object):
    """성능 평가를 위한 evaluator의 베이스 클래스."""

    @abstractproperty
    def worst_score(self):
        """
        최저 성능 점수.
        :return float.
        """
        pass

    @abstractproperty
    def mode(self):
        """
        점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부. 'max'와 'min' 중 하나.
        e.g. 정확도, AUC, 정밀도, 재현율 등의 경우 'max',
             오류율, 미검률, 오검률 등의 경우 'min'.
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        실제로 사용할 성능 평가 지표.
        해당 함수를 추후 구현해야 함.
        :param y_true: np.ndarray, shape: (N, num_classes).
        :param y_pred: np.ndarray, shape: (N, num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다 우수한지 여부를 반환하는 함수.
        해당 함수를 추후 구현해야 함.
        :param curr: float, 평가 대상이 되는 현재 성능 점수.
        :param best: float, 현재까지의 최고 성능 점수.
        :return bool.
        """
        pass
```
Evaluator 클래스는, evaluator를 서술하는 베이스 클래스입니다. 이는 worst_score, mode 프로퍼티(property)와 score, is_better 함수로 구성되어 있습니다. 성능 평가 척도에 따라 ‘최저’ 성능 점수와 ‘점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지’ 등이 다르기 때문에, 이들을 명시하는 부분이 각각 worst_score와 mode입니다.

한편 score 함수는 테스트용 데이터셋의 실제 레이블 및 이에 대한 모델의 예측 결과를 받아, 지정한 성능 평가 척도에 의거하여 성능 점수를 계산하여 반환합니다. is_better 함수는 현재의 평가 성능과 현재까지의 ‘최고’ 성능을 서로 비교하여, 현재 성능이 최고 성능보다 더 우수한지 여부를 bool 타입으로 반환합니다.


#### AccuracyEvaluator 클래스
```python
class AccuracyEvaluator(Evaluator):
    """정확도를 평가 척도로 사용하는 evaluator 클래스."""

    @property
    def worst_score(self):
        """최저 성능 점수."""
        return 0.0

    @property
    def mode(self):
        """점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부."""
        return 'max'

    def score(self, y_true, y_pred):
        """정확도에 기반한 성능 평가 점수."""
        return accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    def is_better(self, curr, best, **kwargs):
        """
        상대적 문턱값을 고려하여, 현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다 우수한지 여부를 반환하는 함수.
        :param kwargs: dict, 추가 인자.
            - score_threshold: float, 새로운 최적값 결정을 위한 상대적 문턱값으로,유의미한 차이가 발생했을 경우만을 반영하기 위함.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps
```
AccuracyEvaluator 클래스는 정확도를 평가 척도로 삼는 evaluator로, Evaluator 클래스를 구현(implement)한 것입니다. score 함수에서 정확도를 계산하기 위해, scikit-learn 라이브러리에서 제공하는 sklearn.metrics.accuracy_score 함수를 불러와 사용하였습니다. 한편 is_better 함수에서는 두 성능 간의 단순 비교를 수행하는 것이 아니라, 상대적 문턱값(relative threshold)을 사용하여 현재 평가 성능이 최고 평가 성능보다 지정한 비율 이상으로 높은 경우에 한해 True를 반환하도록 하였습니다.

<br/>

## (3) 러닝 모델: AlexNet
러닝 모델로는 앞서 언급한 대로 컨볼루션 신경망인 AlexNet을 사용합니다. 이 때, 러닝 모델을 사후적으로 수정하거나 혹은 새로운 구조의 러닝 모델을 추가하는 상황에서의 편의를 고려하여, 컨볼루션 신경망에서 주로 사용하는 층(layers)들을 생성하는 함수를 미리 정의해 놓고, 일반적인 컨볼루션 신경망 모델을 표현하는 베이스 클래스를 먼저 정의한 뒤 이를 AlexNet의 클래스가 상속받는 형태로 구현하였습니다.

## models.layers 모듈
models.layers 모듈에서는, 컨볼루션 신경망에서 주로 사용하는 컨볼루션 층(convolutional layer), 완전 연결 층(fully-connected layer) 등을 함수 형태로 정의하였습니다.

```python
def weight_variable(shape, stddev=0.01):
    """
    새로운 가중치 변수를 주어진 shape에 맞게 선언하고,
    Normal(0.0, stddev^2)의 정규분포로부터의 샘플링을 통해 초기화함.
    :param shape: list(int).
    :param stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
    :return weights: tf.Variable.
    """
    weights = tf.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights


def bias_variable(shape, value=1.0):
    """
    새로운 바이어스 변수를 주어진 shape에 맞게 선언하고, 
    주어진 상수값으로 추기화함.
    :param shape: list(int).
    :param value: float, 바이어스의 초기화 값.
    :return biases: tf.Variable.
    """
    biases = tf.get_variable('biases', shape, tf.float32,
                             tf.constant_initializer(value=value))
    return biases


def conv2d(x, W, stride, padding='SAME'):
    """
    주어진 입력값과 필터 가중치 간의 2D 컨볼루션을 수행함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param W: tf.Tensor, shape: (fh, fw, ic, oc).
    :param stride: int, 필터의 각 방향으로의 이동 간격.
    :param padding: str, 'SAME' 또는 'VALID',
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :return: tf.Tensor.
    """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, side_l, stride, padding='SAME'):
    """
    주어진 입력값에 대해 최댓값 풀링(max pooling)을 수행함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, 풀링 윈도우의 한 변의 길이.
    :param stride: int, 풀링 윈도우의 각 방향으로의 이동 간격. 
    :param padding: str, 'SAME' 또는 'VALID',
                         풀링 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :return: tf.Tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, side_l, stride, out_depth, padding='SAME', **kwargs):
    """
    새로운 컨볼루션 층을 추가함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, 필터의 한 변의 길이.
    :param stride: int, 필터의 각 방향으로의 이동 간격.
    :param out_depth: int, 입력값에 적용할 필터의 총 개수.
    :param padding: str, 'SAME' 또는 'VALID',
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :param kwargs: dict, 추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함.
        - weight_stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
        - biases_value: float, 바이어스의 초기화 값.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_depth = int(x.get_shape()[-1])

    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weights_stddev)
    biases = bias_variable([out_depth], value=biases_value)
    return conv2d(x, filters, stride, padding=padding) + biases


def fc_layer(x, out_dim, **kwargs):
    """
    새로운 완전 연결 층을 추가함.
    :param x: tf.Tensor, shape: (N, D).
    :param out_dim: int, 출력 벡터의 차원수.
    :param kwargs: dict, 추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함. 
        - weight_stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
        - biases_value: float, 바이어스의 초기화 값.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim], stddev=weights_stddev)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases

```
AlexNet의 경우 처음 가중치(weight)와 바이어스(bias)를 초기화(initialize)할 때 각기 다른 방법으로 합니다.

- 가중치: 지정한 표준편차(standard deviation)를 가지는 정규 분포(Normal distribution)으로부터 가중치들을 랜덤하게 샘플링하여 초기화함
- 바이어스: 지정한 값으로 초기화함

이를 반영하고자 weight_variable 함수에서는 가중치를 샘플링할 정규 분포의 표준편차인 stddev을, bias_variable 함수에서는 바이어스를 초기화할 값인 value를 인자로 추가하였습니다. AlexNet의 각 층에 따라 초기화에 사용할 가중치의 표준편차 및 바이어스 값 등이 다르게 적용되기 때문에, 이를 조정할 수 있도록 구현하였습니다.

### models.nn 모듈
models.nn 모듈은, 컨볼루션 신경망을 표현하는 클래스를 담고 있습니다.

#### ConvNet 클래스
```python
class ConvNet(object):
    """컨볼루션 신경망 모델의 베이스 클래스."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        모델 생성자.
        :param input_shape: tuple, shape (H, W, C) 및 값 범위 [0.0, 1.0]의 입력값.
        :param num_classes: int, 총 클래스 개수.
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(tf.float32, [None] + [num_classes])
        self.is_train = tf.placeholder(tf.bool)

        # 모델과 손실 함수 정의
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        모델 생성.
        해당 함수를 추후 구현해야 함. 
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성.
        해당 함수를 추후 구현해야 함. 
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        주어진 데이터셋에 대한 예측을 수행함.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, 예측 과정에서 구체적인 정보를 출력할지 여부.
        :param kwargs: dict, 예측을 위한 추가 인자.
            - batch_size: int, 각 반복 회차에서의 미니배치 크기.
            - augment_pred: bool, 예측 과정에서 데이터 증강을 수행할지 여부.
        :return _y_pred: np.ndarray, shape: (N, num_classes).
        """
        batch_size = kwargs.pop('batch_size', 256)
        augment_pred = kwargs.pop('augment_pred', True)

        if dataset.labels is not None:
            assert len(dataset.labels.shape) > 1, 'Labels must be one-hot encoded.'
        num_classes = int(self.y.get_shape()[-1])
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # 예측 루프를 시작함
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps*batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False,
                                      augment=augment_pred, is_train=False)
            # if augment_pred == True:  X.shape: (N, 10, h, w, C)
            # else:                     X.shape: (N, h, w, C)

            # 예측 과정에서 데이터 증강을 수행할 경우,
            if augment_pred:
                y_pred_patches = np.empty((_batch_size, 10, num_classes),
                                          dtype=np.float32)    # (N, 10, num_classes)
                # 10종류의 patch 각각에 대하여 예측 결과를 산출하고,
                for idx in range(10):
                    y_pred_patch = sess.run(self.pred,
                                            feed_dict={self.X: X[:, idx],    # (N, h, w, C)
                                                       self.is_train: False})
                    y_pred_patches[:, idx] = y_pred_patch
                # 이들 10개 예측 결과의 평균을 산출함
                y_pred = y_pred_patches.mean(axis=1)    # (N, num_classes)
            else:
                # 예측 결과를 단순 산출함
                y_pred = sess.run(self.pred,
                                  feed_dict={self.X: X,
                                             self.is_train: False})    # (N, num_classes)

            _y_pred.append(y_pred)
        if verbose:
            print('Total evaluation time(sec): {}'.format(time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)    # (N, num_classes)

        return _y_pred

```
ConvNet 클래스는, 컨볼루션 신경망 모델 객체를 서술하는 베이스 클래스입니다. 어떤 컨볼루션 신경망을 사용할 것이냐에 따라 그 아키텍처(architecture)가 달라질 것이기 때문에, ConvNet 클래스의 자식 클래스에서 이를 _build_model 함수에서 구현하도록 하였습니다. 한편 컨볼루션 신경망을 학습할 시 사용할 손실 함수(loss function) 또한 ConvNet의 자식 클래스에서 _build_loss 함수에 구현하도록 하였습니다.

predict 함수는, DataSet 객체인 dataset을 입력받아 이에 대한 모델의 예측 결과를 반환합니다. 이 때, 테스트 단계에서의 데이터 증강 방법을 채택할 경우(augment_pred == True), 앞서 설명했던 방식대로 하나의 이미지 당 총 10개의 패치를 얻으며, 이들 각각에 대한 예측 결과를 계산하고 이들의 평균을 계산하는 방식으로 최종적인 예측을 수행하게 됩니다.

#### AlexNet 클래스
```python
class AlexNet(ConvNet):
    """AlexNet 클래스."""

    def _build_model(self, **kwargs):
        """
        모델 생성.
        :param kwargs: dict, AlexNet 생성을 위한 추가 인자.
            - image_mean: np.ndarray, 평균 이미지: 이미지들의 각 입력 채널별 평균값, shape: (C,).
            - dropout_prob: float, 완전 연결 층에서 각 유닛별 드롭아웃 수행 확률.
        :return d: dict, 각 층에서의 출력값들을 포함함.
        """
        d = dict()    # 각 중간층에서의 출력값을 포함하는 dict.
        X_mean = kwargs.pop('image_mean', 0.0)
        dropout_prob = kwargs.pop('dropout_prob', 0.0)
        num_classes = int(self.y.get_shape()[-1])

        # Dropout을 적용할 층들에서의 각 유닛별 '유지' 확률
        keep_prob = tf.cond(self.is_train,
                            lambda: 1. - dropout_prob,
                            lambda: 1.)

        # input
        X_input = self.X - X_mean    # 기존 입력값으로부터 평균 이미지를 뺌

        # conv1 - relu1 - pool1
        with tf.variable_scope('conv1'):
            d['conv1'] = conv_layer(X_input, 11, 4, 96, padding='VALID',
                                    weights_stddev=0.01, biases_value=0.0)
            print('conv1.shape', d['conv1'].get_shape().as_list())
        d['relu1'] = tf.nn.relu(d['conv1'])
        # (227, 227, 3) --> (55, 55, 96)
        d['pool1'] = max_pool(d['relu1'], 3, 2, padding='VALID')
        # (55, 55, 96) --> (27, 27, 96)
        print('pool1.shape', d['pool1'].get_shape().as_list())

        # conv2 - relu2 - pool2
        with tf.variable_scope('conv2'):
            d['conv2'] = conv_layer(d['pool1'], 5, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv2.shape', d['conv2'].get_shape().as_list())
        d['relu2'] = tf.nn.relu(d['conv2'])
        # (27, 27, 96) --> (27, 27, 256)
        d['pool2'] = max_pool(d['relu2'], 3, 2, padding='VALID')
        # (27, 27, 256) --> (13, 13, 256)
        print('pool2.shape', d['pool2'].get_shape().as_list())

        # conv3 - relu3
        with tf.variable_scope('conv3'):
            d['conv3'] = conv_layer(d['pool2'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.0)
            print('conv3.shape', d['conv3'].get_shape().as_list())
        d['relu3'] = tf.nn.relu(d['conv3'])
        # (13, 13, 256) --> (13, 13, 384)

        # conv4 - relu4
        with tf.variable_scope('conv4'):
            d['conv4'] = conv_layer(d['relu3'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv4.shape', d['conv4'].get_shape().as_list())
        d['relu4'] = tf.nn.relu(d['conv4'])
        # (13, 13, 384) --> (13, 13, 384)

        # conv5 - relu5 - pool5
        with tf.variable_scope('conv5'):
            d['conv5'] = conv_layer(d['relu4'], 3, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv5.shape', d['conv5'].get_shape().as_list())
        d['relu5'] = tf.nn.relu(d['conv5'])
        # (13, 13, 384) --> (13, 13, 256)
        d['pool5'] = max_pool(d['relu5'], 3, 2, padding='VALID')
        # (13, 13, 256) --> (6, 6, 256)
        print('pool5.shape', d['pool5'].get_shape().as_list())

        # 전체 feature maps를 flatten하여 벡터화
        f_dim = int(np.prod(d['pool5'].get_shape()[1:]))
        f_emb = tf.reshape(d['pool5'], [-1, f_dim])
        # (6, 6, 256) --> (9216)

        # fc6
        with tf.variable_scope('fc6'):
            d['fc6'] = fc_layer(f_emb, 4096,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu6'] = tf.nn.relu(d['fc6'])
        d['drop6'] = tf.nn.dropout(d['relu6'], keep_prob)
        # (9216) --> (4096)
        print('drop6.shape', d['drop6'].get_shape().as_list())

        # fc7
        with tf.variable_scope('fc7'):
            d['fc7'] = fc_layer(d['drop6'], 4096,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu7'] = tf.nn.relu(d['fc7'])
        d['drop7'] = tf.nn.dropout(d['relu7'], keep_prob)
        # (4096) --> (4096)
        print('drop7.shape', d['drop7'].get_shape().as_list())

        # fc8
        with tf.variable_scope('fc8'):
            d['logits'] = fc_layer(d['relu7'], num_classes,
                                weights_stddev=0.01, biases_value=0.0)
        # (4096) --> (num_classes)

        # softmax
        d['pred'] = tf.nn.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성.
        :param kwargs: dict, 정규화 항을 위한 추가 인자.
            - weight_decay: float, L2 정규화 계수.
        :return tf.Tensor.
        """
        weight_decay = kwargs.pop('weight_decay', 0.0005)
        variables = tf.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables])

        # 소프트맥스 교차 엔트로피 손실 함수
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        softmax_loss = tf.reduce_mean(softmax_losses)

        return softmax_loss + weight_decay*l2_reg_loss
```
<br/>


## 학습 수행 및 테스트 결과
train.py 스크립트에서는 실제 학습을 수행하는 과정을 구현하였으며, test.py 스크립트에서는 테스트 데이터셋에 대하여 학습이 완료된 모델을 테스트하는 과정을 구현하였습니다.

### train.py 스크립트
```python
""" 1. 원본 데이터셋을 메모리에 로드하고 분리함 """
root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')    # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# 원본 학습+검증 데이터셋을 로드하고, 이를 학습 데이터셋과 검증 데이터셋으로 나눔
X_trainval, y_trainval = dataset.read_asirra_subset(trainval_dir, one_hot=True)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.2)    # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

# 중간 점검
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())
print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())
print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())


""" 2. 학습 수행 및 성능 평가를 위한 하이퍼파라미터 설정 """
hp_d = dict()
image_mean = train_set.images.mean(axis=(0, 1, 2))    # 평균 이미지
np.save('/tmp/asirra_mean.npy', image_mean)    # 평균 이미지를 저장
hp_d['image_mean'] = image_mean

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['num_epochs'] = 300

hp_d['augment_train'] = True
hp_d['augment_pred'] = True

hp_d['init_learning_rate'] = 0.01
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: 정규화 관련 하이퍼파라미터
hp_d['weight_decay'] = 0.0005
hp_d['dropout_prob'] = 0.5

# FIXME: 성능 평가 관련 하이퍼파라미터
hp_d['score_threshold'] = 1e-4


""" 3. Graph 생성, session 초기화 및 학습 시작 """
# 초기화
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

sess = tf.Session(graph=graph, config=config)
train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)
```

<br/>




## Reference
[1] 이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기 [[url]](http://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html) <br/>

```python
```

<br/>