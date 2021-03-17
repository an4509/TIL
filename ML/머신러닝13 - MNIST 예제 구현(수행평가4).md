# MNIST 예제 구현(수행평가4)

* tensorflow 구현
* MNIST는 vision data set (손으로 쓰여진 숫자 이미지)
* 이미지는 원래 3차원(가로,세로,컬러).  차원 잘 맞춰줘야함.
* 흑백은 2차원
* 2차원 데이터가 여러개 있기 때문에 3차원인데 편하게 하기 위해
* 2차원 데이터를 1차원으로 변환(ravel)
* label은 그 데이터의 숫자(0~9 총 10개)
* 그리고 모델이 완성되면 손글씨로 숫자 써서 사진 찍어 컴퓨터에 업로드
* 이미지를 dats set으로 변환해서(어려움)  prediction
* 90 초반대로 정확도가 나올거임
* 먼저 matplotlib으로 그려보기(이미지)



## 필요한 import 가져오기

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # 정규화 
from sklearn.model_selection import train_test_split  # train 데이터와 test 데이터 나누기
from sklearn.model_selection import KFold  # KFold를 해주는 함수
from sklearn.metrics import confusion_matrix, classification_report  # confusion matrics를 보여주는 함수
```



## 데이터 가져오기

```python
# Raw Data Loading
df = pd.read_csv('./data/mnist/train.csv')
display(df.head(), df.shape)
```

* head()로 상위 5개만 보기, shape 확인해서 열, 행 개수 확인.
*  Kaggle에서 데이터 명세 확인도 해보기!
* 각 픽셀의 값은 0~255 사이의 값 확인.



## 이상치 & 결측치

* 픽셀의 값이기 때문에 이상치, 결측치가 없음.



## 이미지 확인해보기

```python
# 이미지 확인
img_data = df.drop('label', axis=1, inplace=False).values

fig = plt.figure()
fig_arr = []   # 10개의 subplot을 만들고 그 각각의 subplot을 list에 저장

for n in range(10):
    fig_arr.append(fig.add_subplot(2,5,n+1))
    fig_arr[n].imshow(img_data[n].reshape(28,28),
                     cmap='Greys',
                     interpolation='nearest')
plt.tight_layout()
plt.show()
```

* label 열 drop해서 values로 2차원 배열로 만들어주기
* 10개 정도 이미지를 확인하기 위해 fig_arr로 subplot담을 리스트 만들기.
* for문으로 10개 이미지 append 해주기.
* img_data[n]으로 행 indexing 후 reshape으로 28*28픽셀로 변환
* cmap으로 흑백(Greys)로 설정해줘야 2차원을 읽어줌.

