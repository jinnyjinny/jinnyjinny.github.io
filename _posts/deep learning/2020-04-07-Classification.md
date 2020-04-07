---
layout: post
title: Image Classification 실습해보기 (AlexNet으로 개vs고양이 분류)
category: deep learning
tags: [classification, alexnet, tensorflow]
comments: true

---

### 이미지 인식 문제를 위한 딥러닝의 기본 요소
- 데이터셋
- 성능 평가
- (딥)러닝 모델
- (딥)러닝 알고리즘

그러면 지금부터, 위에서 언급한 딥러닝의 4가지 기본 요소를 기준으로 삼아, ‘개vs고양이 분류’ 문제 해결을 위해 직접 제작한 AlexNet 구현체를 소개해 드리도록 하겠습니다.

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

## (3) 러닝 모델: AlexNet
러닝 모델로는 앞서 언급한 대로 컨볼루션 신경망인 AlexNet을 사용합니다. 이 때, 러닝 모델을 사후적으로 수정하거나 혹은 새로운 구조의 러닝 모델을 추가하는 상황에서의 편의를 고려하여, 컨볼루션 신경망에서 주로 사용하는 층(layers)들을 생성하는 함수를 미리 정의해 놓고, 일반적인 컨볼루션 신경망 모델을 표현하는 베이스 클래스를 먼저 정의한 뒤 이를 AlexNet의 클래스가 상속받는 형태로 구현하였습니다.

## models.layers 모듈
models.layers 모듈에서는, 컨볼루션 신경망에서 주로 사용하는 컨볼루션 층(convolutional layer), 완전 연결 층(fully-connected layer) 등을 함수 형태로 정의하였습니다.

-- 이어서 계속 -- 

## Reference
[1] 이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기 [[url]](http://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html) <br/>