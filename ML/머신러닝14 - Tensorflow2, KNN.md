# Tensorflow 2.x

* 구글이 만든 Deep Learning 용 Library.
* 2.x 버전으로 넘어오면서 많은 변경점이 생김.
* Keras가 안으로 들어옴.
  * 프랑소와 숄레라는 사람이 개발했으나 구글에 입사하여 keras가 tensorflow가 포함됨.
* Eager Execution (즉식 실행) -> 코드의 직관성이 높아짐.
* Keras가 High-level API로 공식적으로 지원.



## 새로운 가상환경 생성

```
conda create -n data_env_tf2 python=3.7 openssl

conda activate data_env_tf2
conda install nb_conda # 주피터 노트북
conda install numpy
conda install pandas
conda install tensorflow
conda install matplotlib
pip install sklearn
```



## new 만들기

* new 버튼을 누르면 새로 만든 가상환경이 보임!!



## 버전확인

```python
import tensorflow as tf
print(tf.__version__)
```



## 달라진 점

```python
# 1. 초기화 코드가 삭제 됨!
# 1.15 버전은 
# session이 있어야 함.
# sess.run(tf.global_varibales_initializer())

# 2. session이 삭제됨!
# 2.x버전에서는
# sess가 객체를 만들어서 실행시키지 않아도 됨.
print(W.numpy()) # numpy 함수를 이용하면 numpy값으로 떨어짐. # eager execution.

# 3. placeholder를 더이상 사용하지 않음.
# 그래프르 그려서 그래프 안에 값을 집어 넣는 방식 -> python 스타일로 바뀜
a = 50
b = 70

def my_sum(x,y):
    t1 = tf.convert_to_tensor(x)  # ()안에 들어오는 값을 tensor로 바꿔서 t1을 node로 변경 가능
    t2 = tf.convert_to_tensor(y)
    
    return t1 + t2

result = my_sum(a,b)
print('결과는 : {}'.format(result.numpy())) # 일반 프로그램 방식으로 돌아옴.

# 이 3가지들이 코드의 직관성이 높아짐 (eager execution)
```





# Machine Learning의 raise and fall(흥망성쇠)

* 1986년 제프리 힌트 (back propagation) 오차 역전파 알고리즘에 의해 Machine Learning 기법 중 Neural Network 부분(Deep Learning)이 다시 각광
* 2010년 부터 본격적으로 Deep Learning이 발전 시작.
  * Deep Learning Library가 만들어지기 시작함.
* 2010 벤지오의 Theano Library를 시작으로 발전이 시작 됨. (Theano - 역사에 기록된 최초의 여성 수학자, 피타고라스 아내)
* 2013 버클리 대학의 중국 유학생 가양청의 caffe
* 2014 로난의 touch3
  * 다양한 deep learning libary가 매년 개발되고 발전되고 있던 상황이었음
* 2015년에는 갑자기 쓸만한 library가 나옴 
  * 카네기멜로대학 MXnet, google Tensorflow, 프랑소와 숄레 keras
* 2016 MS의 CNTK 나옴.
* 2017 Theano 개발중지 선언
* 2019 tensorflow가 2.0으로 개발됨.



# 정리하기

* tensorflow 2.x 버전으로구현하기 전에 간단하게 정리해보기



## Simple Linear Regression

* 독립변수가 1개
* Y = WX + b
* 예측값과 t와 차이를 loss로 줄이기



## Multiple Linear Regression

* 독립변수가 2개 이상
* Y = WX + b
* 예측값과 t와 차이를 loss로 줄이기



## Multiple Logistic Regression

* 독립변수가 여러개
* t가 이진법 (Binary  Classification)
* sigmoid 처리 후 정답값과 비교
* loss처리 -> cross entropy



## Multinomial classification

* 독립변수가 여러개
* t가 3개 이상
* t를 one-hot encoding 처리
* softmax 처리
* cross entropy



# Logistic Regression을 Keras의 Model로 표현하기

![](md-images/keras%20-%20Logistic%20regression.PNG)

## Model

* Keras의 Model은 전체 layer들을 포함하고 있는 box
* input과 output이 model을 거쳐서 나오게 됨.
* 내부적으로 W가 생성되어 포함되어 있음
* input -> output 이동시 W 행렬곱이 되어 이동.

* 이후 output에서 sigmoid 처리되어 값이 도출.
* 값이도출되어 cross entropy로 t값과 비교
* 이러한 model을 keras에서 만들어야 함.



## Layer

* layer은 층으로 표현되고 여러개 model 안에 존재.
* layer은 종류가 있음
  * input layer : model 안에 데이터가 들어오면 받아주는 layer
  * output layer : 계산된 결과를 model 밖으로 뽑아내는 layer
  * hidden layer : 다 계층이 존재. deep learning 시 존재. Logistic Regression에는 없다!



## Input

* input은 training data set
* 만일 Multinomial 일 때 input layer안에 logistic을 t의 개수만큼 담고 있음



## Output

* output layer에서도 logistic 만큼 연산공간이 있음.

* active function이 포함되어 있음.



## Keras 코드로 표현하기



### Import

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential # model box 그리기
from tensorflow.keras.layers import Flatten, Dense 
# layer 중 Flatten(input layer 역할) 다차원을 1차원으로 변경,
# Dense(output layer 역할) 
from tensorflow.kears.optimizers import SGD, Adam
# SGD, Adam 는 알고리즘(optimizer)
```



### model box 만들기

```python
model = Sequential()  # model box 생성
```



### model layer 만들기

```python
odel.add(Flatten(input_shape(1,)))  #  input역할을 할 Flatten layer 추가
model.add(Dense(1, activation='sigmoid')) # output 역할을 할 Dense layer 추가
# 가로 안 숫자는 logistic 개수
# 숫자 뒤에 logistic들이 어떤 activation을 쓰는지 명시
```



### 모델 loss, optimizer 지정 (compile)

```python

model.compile(optimizer=SGD(learning_rate=1e-3),
              loss='mse') # optimizer 선정
```



### 모델 학습 (fit)

```python
model.fit(x_data_train, # model 안에 데이터 밀어넣기
          t_data_train,
          epochs=10,
          batch_size=200,
          validation_split=0.2)
```



### 모델 평가 (evaluate)

```python
model.evaluate(x_data_test,t_data_test)
```



### 모델 예측(predict)

```python
model.predict(x_data_predict)
```



### 저장 및 불러오기

```python
model.save('./myModel.h5') # 학습이 끝난 모델을 file로 저장할 수 있음.
# 불러오기
model = tf.keras.models.load_model('./myModel.h5')
```





# Linear Regression Keras로 표현하기

* ozone 예제로 구현해보기
* sklearn과 tensorflow 2.x버전으로 구현
* 이번에는 결측치와 이상치 처리를 단순히 삭제하지 않고 적절한 값으로 대치



## 결측치 처리

1. Deletion (결측치 제거)

   * 결측치 제거안에서도 Listwise 삭제 방식과 Pairwise 삭제 방식이 있음.

     1-1. 손쉽게 접근하는 방법이 Listwise 삭제 방식

     * NaN이 존재하면 행 자체를 삭제 (우리가 했던 방식)

     * 단점은 NaN을 제외한 col의 다른 의미있는 데이터가 같이 삭제됨.

     * 데이터가 충분히 많고 Nan의 빈도가 상대적으로 작을 경우 최상의 방법.

       

     1-2. Pairwise

     * 의미있는 데이터가 삭제되는걸 막기 위해 행 전체를 삭제하지 않음.
     *  그 값만 모든 처리에서 제외(오히려 문제가 발생할 여지가 있음.)
     *  특별한 경우에만 사용.

   

2. Imputation(결측치 보간)

   * 결측치 보간 기법에는 크게 2가지 방식이 있음.
        2-1. 평균화 기법

     * 전체 데이터의 평균을 내어 값을 채우는 것. 일반적으로 많이 사용.
     * 평균(mean), 중앙값(median), 최빈값(mode)
     * 장점 : 쉽고, 빠름.
     * 단점 : 통계분석에 영향을 많이 미침.

     

        2-2. 예측 기법

     * 결측치가 종속변수일 때 사용.

     * 결측치들이 완전히 무작위적으로 관찰되지 않았다는 것을 가정.

   



* 우리 예제에서는 머신러닝 기법 중 Regression이 아닌 KNN을 이용해서 imputation 진행
* 일반적으로 평균화 기법보다는 조금 더 나은 결측치 보간 가능.
* 그러면 결측치 보간을 위해서 KNN부터 알아보기





# KNN(K-Nearest Neighbor, K 최근접이웃)

* KNN 알고리즘은 K 최근접이웃.

![](/md-images/KNN.PNG)



## 특징

* 장점
  * 상당히 간단한 모델
  * 학습데이터가 많으면 꽤 정확한 값을 도출
* 단점
  
* 각각 데이터의 거리를 계산하느라 시간이 오래걸림.
  
* 반드시 정규화 진행해야함.

* classcification 기준으로 설명하자면,

  * K를 기준으로 동심원(이웃이 가장 가까운 이웃이 K 개 되게끔 포함되는)
  * 동신원 안에 더 많이 포함된 데이터 유형으로 분류.

* Regression 기반으로도 사용 가능.

  * 같은 원리로, label 값을 동심원 기준으로 가장 가까운 이웃 k개 평균을 내서 예측

* KNN은 학습이라는 절차가 필요 없음.

  * 새로운 데이터가 들어왔을 떄 기존 data들과의 거리를 계산해서 예측을 수행

    

## KNN의 두개의 hyperparameter

* 이웃의 수(K)
  * k가 작을 경우 과대적합이 발생할 수 있음.
  * k가 너무 크면 과소적합이 발생할 수 있음.
  * 그래서 단점이 적당한 k값을 결정하고 기준을 잡기가 애매모호.
  * K=1 일 때(1-NN) 오차범위가 이상적인 오차범위의 2배보다 같거나 작음.  수학적 증명 완료. 성능보장
* 거리측정방식
  * Euclidean distance 최단 직선거리
  * Manhattan distance 좌표방향이동
  * Mahalanobis distance  데이터 밀도(분산)를 고려한 거리



## KNN 코드로 알아보기

```python
# KNN을 알아보기.
# sklearn을 이용해서 BMI예제를 가지고 Logistic Regression을 이용한
# accuracy값과 KNN을 이용한 accuracy값을 비교해 보기

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Raw Data Load
df = pd.read_csv('./data/bmi.csv', skiprows=3)
# display(df)

# data split
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(df[['height', 'weight']], df['label'],
                 test_size=0.3, random_state=0)

# 정규화
scaler = MinMaxScaler()
scaler.fit(x_data_train)
x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)

# Logistic Regression
model = LogisticRegression()
model.fit(x_data_train_norm, t_data_train) # sklearn은 one-hot처리를 하지 않음
print(model.score(x_data_test_norm, t_data_test)) # 0.9845
# 비교를 위해서 정규화 된 데이터를 넣은거임! 원래는 안해줘도 됨.


# KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(x_data_train_norm, t_data_train)
print(knn_classifier.score(x_data_test_norm,t_data_test)) # 0.998
```



