# Logistic Regression

> Classification 중 한 종류인 Logistic Regression을 알아보자



* Classification 알고리즘 중에 정확도가 상당히 높은 알고리즘
* Deep Learning의 기본 component로 사용

```python
# Logistic Regression이 어떤 의미인지 알아보기

import warnings
import numpy as np
from sklearn import linear_model
import mglearn # Data Set을 가져오기 위한 Utility module
import pandas as pd
import matplotlib.pyplot as plt

# warning off
warnings.filterwarnings(action='ignore')

# Training Data Set
x,y = mglearn.datasets.make_forge()
# print(x)  # 좌표를 들고 있음. (x축 좌표, y축 좌표)
# print(y)  

# mglearn.discrete_scatter로 확인해보기 (x축의 값, y축의 값, 점의 형태)
mglearn.discrete_scatter(x[:,0], x[:,1], y)

# Linear Regerssion으로 학습
# 가장 잘 표현하는 직선을 그리기.
model = linear_model.LinearRegression()
model.fit(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1))
print(model.coef_) # 직선의 기울기(Weight) : [[-0.17382295]]
print(model.intercept_) # bias : [4.5982984]
plt.plot(x[:,0], x[:,0] * model.coef_.ravel() + model.intercept_)
```





# 지도학습

> 지도학습은 모델의 결과형태에 따라 다시 2가지 형태로 나뉜다.



* 예측값이 연속적인  숫자형태.
* 예측값이 어떤 부류에 속하는지 알려주는 형태.



## Review

* 연속적인 숫자(regression) 형태 지도학습은 classical Linear Regression Model을 이용해서 구현
* 단변량 형태는 Simple Linear Regression.
* 다변량 형태는 Multiple Linear Regresson.



## Classification

> Training Data Set의 특징과 분포를 이용하여 학습한 후 미지의 데이터에 대해서 결과가 어떤 종류의 값으로 분류될 수 있는지 예측하는 작업.
>
> Classification은 분류의 개수에 따라 2가지 개념으로 나뉜다.



* Binary  Classification
  * 분류가 2개 밖에 없을 때
  * ex) 합/불
* Multinomial Classification
  * 여러개 중에 분류 예측할 때



* 그러면, Linear Regression으로는 할 수 없나?
  * 정확하지 않은 prediction이 됨.
  * 직선이 1을 넘기게 되어 모순됨.

```python
# logistic regression대신 linear regression을 이용해도 되나?
# 왜 안되는지 살펴보자
# 데이터의 의미


import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Training Data Set 준비
x_data = np.array([1, 2, 5, 8, 10, 30])  # 공부시간 / 30은 지대점
# 데이터의 의미를 몰라서 도메인 관리자에게 이상치 유무를 확인해야함.
# 

t_data = np.array([0, 0, 0, 1, 1, 1])  # 합격여부(0: Fali, 1 : Pass)

model = linear_model.LinearRegression()
model.fit(x_data.reshape(-1, 1),
          t_data.reshape(-1, 1))
print('기울기 : {}, 절편 ; {}'.format(model.coef_, model.intercept_))

plt.scatter(x_data, t_data)
plt.plot(x_data, x_data * model.coef_.ravel() + model.intercept_)
plt.show()

print(model.predict([[7]]))  # [[0.41831972]] => 0.5보다 작기 때문에 fail

# 그러면 시험에 통과하려면 적어도 몇시간 공부해야 하나?
print((0.5 - model.intercept_) / model.coef_.ravel())  # [9.33333333]

# training data에는 8시간 공부하면 합격된다고 되있었지만,
# 실제 데이터를 이상치와 근접한 값을 넣어줬더니
# 결과가 모순되게 나옴. 결론은 이러한 문제 때문에 Linear Regression으로는 
# 판단문제를 해결할 수 없음.
```



## 절차

> 직선인 Linear Regression을 sigmoid 해서 곡선 형태로 변경

* training data set
* Logistic Regression(Model)
  * Linear Regression (Wx + b)
  * Classification (y = sigmoid(Wx + b))



### Sigmoid

* sigma(x) = 1 / 1 + e^ -x
* x축 값이 아무리 작아져도 0이하로 떨어지지 않고, 아무리 커져도 1보다 커지지 않음.
* 0과 1사이에 S자 모양의 그래프가 그려짐



* Python으로 구현하기

```python
# sigmoid 함수의 형태를 살펴보기

import numpy as np
import matplotlib.pyplot as plt

x_data = np.arange(-7,8)
sigmoid_t_data = 1 / (1 + np.exp(-1 * x_data))

print(plt.plot(x_data, sigmoid_t_data))
```





### Model 식

> x에 Linear Regression 식인 Wx + b를 대입

* Model = 1 / 1 + e^ -(Wx + b)



## Linear Regression의 출력(Model)

* Wx + b가 어떠한 값을 가지더라도 추력함수로 sigmoid 함수를 이용하면,
* 0 ~ 1 사이의 실수값이 도출하고 0.5 이상이면 1을 출력, 0.5 이하면 0으로 출력.
* 최적의 W와 b를 구하기 위해 loss함수를 정의 후 반복적으로 미분화하면서 W,b갱신
* Model을 적용한 Loss 함수를 구하면 구부러진 그래프가 구해져 불편.. 직선그래프가 필요
* Logistic Regression의 새로 정의한 loss 함수가 필요. (Cross Entropy)

![](md-images/not%20convex.PNG)





## Cross Entropy (Log Loss)



* Model이 1 / 1 + e^-(Wx+b) 일 때,  아래 그림과 같다.

![](md-images/cross%20entropy.PNG)

```python
def loss_func(input_obj):  # W와 b가 입력으로 들어가야 함.
    input_W = input_obj[0].reshape(-1,1)
    input_b = input_obj[1]
    
    # linear regrssion의 hypothesis
    z = np.dot(x_data, input_W) + input_b  # Wx + b
    # logistic regression의 hypothesis
    y = 1 / (1 + np.exp(-1 * z)) 
    
    # log 연산 시 무한대로 발산하는 것을 방지하기 위한 수치처리방식
    delta = 1e-7
    
    # cross entropy
    return -np.sum(t_data*np.log(y+delta) + (1-t_data)*np.log(1-y+delta))
```



## Predict

```python
# predict
def logistic_predict(x):   # [[13]] => 13시간 공부하면?
    z = np.dot(x,W) + b
    y = 1 / (1 + np.exp(-1 * z))
    
    if y < 0.5:
        result = 0
    else:
        result = 1
    return result, y   # result는 결과값, y는 확률값

study_hour = np.array([[13]])
print(logistic_predict(study_hour))  # 결과 : 1(합격), 확률 : 0.54435052
```





## Tensorflow로 구현하기

```python
# Tensorflow 구현

# training data set
x_data = np.arange(2,21,2)  # 공부시간(독립변수)
t_data = np.array([0,0,0,0,0,0,1,1,1,1])  # 합격여부(14시간부터 1, 종속변수)


#  placeholder
X = tf.placeholder(dtype=tf.float32)
T = tf.placeholder(dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([1,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Hypothesis
logit = W * X + b
H = tf.sigmoid(logit)

# loss function(Cross Entropy)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, 
                                                              labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

# session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(300000):
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss],
                                         feed_dict={X : x_data,
                                                    T : t_data})
    if step % 30000 == 0:
        print('W : {}, b : {}, loss : {}'.format(W_bal, b_val, loss_val))
        
        
study_hour = np.array([[13]])
result = sess.run(H, feed_dict={X : study_hour})
print(result)  # [[0.5771991]]
```



## Sklearn으로 구현하기

```python
# sklearn 으로 구현하기
model = linear_model.LogisticRegression()

model.fit(x_data, t_data.ravel())  # LogisticRegression은 2차원, 1차원

study_hour = np.array([[13]])
print(model.predict(study_hour))  # 결과 : 0(불합격)
result_pro = model.predict_proba(study_hour)  # 
print(result_pro)  # [[0.50009391 0.49990609]] / 떨어질 확률, 합격할 확률
```





# 복습

* linear와 logistic의 차이 알기.
* python 코드 구현 이해.
* 다변수일 때는 어떻게 해야될까?





