---
layout: post
title: K-fold 개념과 Stratified cross validation 적용해보기
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

if sample_size != None and sample_size < set_size:
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

















```














### Reference
[1] 이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기 [[url]](http://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html) <br/>