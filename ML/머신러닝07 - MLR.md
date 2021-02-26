# Multiple Linear Regression

> 독립변수가 여러개



* Classical Linear Regression Model을 기반으로 구현
* y = b0 + bi*S(xi +bi) 인 기본식에서 만약 독립변수가 늘어나면
* H = b0 + b1x1 + b2x2 + b3x3 가 되어  w1, w2, w3을 구해야함.
* 독립변수가 여러개가 되면 직선이 아니라서 그래프를 그릴 수 없음.
* H = wx + b가 아니라, H = w1x1 + w2x2 + w3x3 + b
* 따라서, H = x1w1 + x2w2 + x3w3 + b로 교환을 하고
* H = |x11, x12, x13| * |w1| + b = | y11|

​              |x21, x22, x23|     |w2|          | y21|

​          	|x31, x32, x33|    |w3|           | y31|



* H = X(2차원) * W(2차원) + b



![](md-images/%EB%8F%85%EB%A6%BD%EB%B3%80%EC%88%983%EA%B0%9C.PNG)



## 코드로 구현하기

```python
# Multiple Linear Regression
# 온도, 태양광세기, 바람세기을 이용하여 Ozone량을 예측

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model

# Raw Data Loading
df = pd.read_csv('./ozone.csv')

# 학습에 필요한 데이터부터 추출
training_data = df[['Temp','Wind','Solar.R','Ozone']]
# display(training_data)   # 153 rows × 4 columns

# 결측치 처리
training_data = training_data.dropna(how='any')
# display(training_data) # 111 rows × 4 columns

# 이상치 처리
zscore_threshold = 1.8

for col in training_data.columns:
    tmp = ~(np.abs(stats.zscore(training_data[col])) > zscore_threshold)
    training_data = training_data.loc[tmp]

#display(training_data)   # (86, 4)

# 정규화 처리
scaler_x = MinMaxScaler()  # 객체 생성
scaler_t = MinMaxScaler()  # 객체 생성
scaler_x.fit(training_data[['Temp','Wind','Solar.R']].values)
scaler_t.fit(training_data['Ozone'].values.reshape(-1,1))

training_data_x = scaler_x.transform(training_data[['Temp','Wind','Solar.R']].values)
training_data_t = scaler_t.transform(training_data['Ozone'].values.reshape(-1,1))

# Tensorflow 코드

# placeholder
X = tf.placeholder(shape=[None,3], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Hypothesis
H = tf.matmul(X,W) + b

# loss function
loss = tf.reduce_mean(tf.square(H-T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

# session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(300000):
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss], 
                                         feed_dict={X: training_data_x, 
                                                    T: training_data_t})
    
    if step % 30000 == 0:
        print('W : {}, b : {}, loss : {}'.format(W_val,b_val,loss_val))
```



* 예측값과 sklearn으로 확인해보기

```python
# sklearn을 이용해서 구현해보기

# training Data set
training_data_x = scaler_x.transform(training_data[['Temp', 'Wind', 'Solar.R']].values)
training_data_t = scaler_t.transform(training_data['Ozone'].values.reshape(-1,1))

model = linear_model.LinearRegression()
model.fit(training_data[['Temp', 'Wind', 'Solar.R']].values
          , training_data['Ozone'].values.reshape(-1,1))

# sklearn은 머신러닝내용을 모르는 사람도 데이터를 쉽게 학습해서 
# 예측값을 알아낼 수 있또록 모듈화 시켜서 우리에게 제공!
# 정규화를 빼고 실행해야함.

print('W : {}, b : {}'.format(model.coef_, model.intercept_))
```



```python
# 예측값 구하기

# tensorflow를 이용해서 만든 모델로 예측값을 구하고
# sklearn으로 구현한 모델을 이용해서 예측값을 구해서
# 값을 비교해 보기
# 예측을 할 값은 => (온도, 바람, 태양광세기) => [80, 10, 150]
# 모델데이터는 정규화되어 있기 때문에 맞춰서 예측값을 정규화해서 넣어줘야함!!!!!

# prediction

# sklearn을 이용
sklearn_result = model.predict([[80.0, 10.0, 150.0]])
print(sklearn_result)


# tensorflow를 이용

predict_data = np.array([[80, 10, 150]])
scaled_predict_data = scaler_x.transform(predict_data)
tensorflow_result = sess.run(H, feed_dict={X: scaled_predict_data})

# 이후 정규화를 다시 풀어줘야 한다.
tensorflow_result = scaler_t.inverse_transform(tensorflow_result)

print(tensorflow_result)model = linear_model.LinearRegression()
# model.fit(training_data_x,training_data_t) 가 아님
model.fit(training_data[['Temp','Wind','Solar.R']].values,
         training_data['Ozone'].values.reshape(-1,1))

# sklearn은 머신러닝내용을 모르는 사람도 데이터를 쉽게 학습해서
# 예측값을 알아낼 수 있도록 모듈화 시켜서 우리에게 제공!

# print('W : {}, b : {}'.format(model.coef_, model.intercept_))
print(model.predict([[80.0,10.0,150.0]])) #[[38.8035437]]
```

**답**

```python
[[38.8035437]]
[[39.23277]]
```



**정리**

* 독립변수가 여러개일 때는 코드가 크게 다르지 않음. 종속변수 데이터를 넣어줄 때 데이터의 형태에 주의하자!

* 정규화를 진행했으면 예측을 위한 입력변수도 반드시 정규화를 진행.
* 나온 정답값은 정규화가 되어 나오므로 다시 정규화를 풀어줘야함.
* sklearn은 이러한 작업이 모두 포함되어 정규화 진행이 필요가 없음.