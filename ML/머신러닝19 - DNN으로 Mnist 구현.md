# Deep Learning

* 1개의 Logistic Regression을 node라고 표현.
* Hidden Layer 개수가 많을 수록 deep 하다고 표현.
* DNN과 Deep Learning은 같은 것. 브랜드 이름만 다를 뿐.
* MNIST로 DNN을 구축 해보기 (TF1과 TF2로 구현해보기!!)



## MNIST 구현

* Tensorflow 1.15 버전



### 데이터 전처리

```python
# MINIST DNN 구현(TF 1.15)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Raw Data Loading
df = pd.read_csv('/content/drive/MyDrive/Machine Learning Colab/MNIST/train.csv')
display(df.head(), df.shape)

# 결측치 이상치는 없음

# 이미지 확인
img_data = df.drop('label', axis=1, inplace=False).values


plt.imshow(img_data[3].reshape(28,28), cmap='Greys', interpolation='nearest')
plt.show()

# train, test 데이터 분리
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(df.drop('label', axis=1, inplace=False), df['label'], test_size=0.3, random_state=0)

# Normalization
scaler = MinMaxScaler()
scaler.fit(x_data_train)
x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)

# Tensorflow implementation
sess = tf.Session()
t_data_train_onthot = sess.run(tf.one_hot(t_data_train, depth=10))
t_data_test_onthot = sess.run(tf.one_hot(t_data_test, depth=10))
```



### Tensorflow 1.15



#### Input Layer

```python
# tensorflow graph를 그리기.
# placeholder (input layer)
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
T = tf.placeholder(shape=[None, 10], dtype=tf.float32)
```

* X의 shape은 feature가 784개이기 때문에 [None, 784].
* T의 shape은 labels의 종류가 10개이기 떄문에 [None, 10].



#### Hidden Layer

```python
# hidden layer1
W2 = tf.Variable(tf.random.normal([784, 64])) # 2번째 인자는 node의 개수 즉 몇 개의 logistic을 할것인가 우리가 지정
b2 = tf.Variable(tf.random.normal([64])) # logistic이 512개 이니 그만큼 bias도 필요
layer2 = tf.sigmoid(tf.matmul(X,W2) + b2)

# hidden layer2
W3 = tf.Variable(tf.random.normal([64, 32])) 
b3 = tf.Variable(tf.random.normal([32])) 
layer3 = tf.sigmoid(tf.matmul(layer2,W3) + b3)

# hideen layer3
W4 = tf.Variable(tf.random.normal([32, 16])) 
b4 = tf.Variable(tf.random.normal([16])) 
layer4 = tf.sigmoid(tf.matmul(layer3,W4) + b4)
```

* Variable이 각가의 hidden layer 역할.
* shape은 Input layer에서 들어오는 데이터 개수, 임의의 node의 개수를 지정 .
* 이후 이전 layer에서 들어오는 개수를 맞춰줘야 함.
* sigmoid 함수 포함시켜서 이전 layer와 w의 행렬곱.



#### Ouput Layer

```python
# output layer
W5 = tf.Variable(tf.random.normal([16, 10])) # 최종 output은 onehot encoding의 depth
b5 = tf.Variable(tf.random.normal([10])) 

logit = tf.matmul(layer4,W5) + b5
H = tf.nn.softmax(logit)
```

* output layer에서 나가는 최종 데이터 개수는 onehot encoding의 depth 크기 (labes의 개수)
* H는 결국 우리가 각각의 labels에 대한 확률을 구하기 때문에 sotfmax 함수 사용



#### Train

```python
# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# 초기화
sess.run(tf.global_variables_initializer())

# epochs = 1000
for step in range(1000):
  _, loss_val = sess.run([train, loss], feed_dict={X:x_data_train_norm, T:t_data_train_onthot})

  if step % 100 ==0:
    print('loss : {}'.format(loss_val))
```



#### Accuracy

```python
# Accuracy
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

print('정확도 : {}.'.format(sess.run(accuracy, feed_dict={X:x_data_test_norm, T:t_data_test_onthot})))
```

* 정확도가 70 이하로 나옴

* 학습이 잘 안된다 왜그럴까?



### Machine Learning의 흥망성쇠

* Minsky 가 적절한 학습방식을 찾기 힘들다고 선언
* AI가 망함.
* 1974년 웨어보소가 획기적인 논문을 발표하지만 주목을 받지 못함.
* 1982년 다시 한번 발표하지만 주목을 받지 못함
* 1986년 제프리 힌튼이 재발견 (오차 역전파 , Back propagation)
* AI가 다시 부활
* Back propagation에서 미분을 행렬곱 방식으로 변환
* 우리나라에서도 1993~5년도에 붐이 일어남.
* Back propagation은 역방향으로 w,b를 업데이트
* layer가 많아지면 variacing Gradient 문제가 일어나서 w,b가 원하는 만큼 조절이 안 된다. back propagation 이 완전하지 않다.
* 다시 AI(Neural network) 침체기
* 다시 다른 기법들이 각광을 받음.
* 힌튼 교수가 1990년대에 CIFAR을 설립하고 NN을 연구함
* 초기화와 다른 activation 함수를 사용해야된다는 것을 발견 및 증명
* sigmoid가 아닌 relu를 사용
* 다시 NN이 흥하고 있음.
* 앞으로 우리는 최신의 초기화방법과 activation 함수를 쓸 것임. 



#### 초기화 기법

* Xavier Initialization 

  * 입력의 개수와 출력의 개수를 이용해서 weight의 초기값을 결정하는 방식

  ```python
  W = np.random.randn(num_of_input, num_of_output) / np.sqrt(num_of_input)
  ```

  

```python
# 기존
W2 = tf.Variable(tf.random.normal([784, 64]))

# Xavier Initialization
W2 = tf.get_variable('W222', shape=[784, 64], 
                     initializer=tf.contrib.layers.xavier_initializer())
```



* He`s Initialization

  * Xavier Initialization의 확장버전

  ```python
  W = np.random.randn(num_of_input, num_of_output) / np.sqrt(num_of_input / 2)
  ```

```python
# 기존
W2 = tf.Variable(tf.random.normal([784, 64]))

# He`s Initialization
W2 = tf.get_variable('W2', shape=[784, 64], initializer=tf.contrib.layers.variance_scaling_initializer())
```



####  Relu activation 함수

* Sigmoid 대체 함수. 자세한 설명은 뒤쪽에 설명

  ```python
  # 기존 
  layer2 = tf.sigmoid(tf.matmul(X,W2) + b2) 
  
  # relu
  layer2 = tf.nn.relu(tf.matmul(X,W2) + b2)
  ```

  

#### 전체 코드 

```python
# tensorflow graph를 그리기.
# placeholder (input layer)
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
T = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# Weight & bias
# hidden layer1
# weight의 초기값을 랜덤으로 하면 안되요! 특정한 방법을 이용해야 좋은 성능을 얻을 수 있음
# 1. Xavier Initialization : 입력의 개수와 출력의 개수를 이용해서 weight의 초기값을 결정하는 방식
# W = np.random.randn(num_of_input, num_of_output) / np.sqrt(num_of_input)

# 2. He`s Initialization : Xavier Initialization의 확장버전
# W = np.random.randn(num_of_input, num_of_output) / np.sqrt(num_of_input / 2)

# W2 = tf.Variable(tf.random.normal([784, 64])) # 2번째 인자는 node의 개수 즉 몇 개의 logistic을 할것인가 우리가 지정
W2 = tf.get_variable('W222', shape=[784, 64], 
                     initializer=tf.contrib.layers.xavier_initializer())

# He`s 초기법
# W2 = tf.get_variable('W2', shape=[784, 64], initializer=tf.contrib.layers.variance_scaling_initializer())

b2 = tf.Variable(tf.random.normal([64])) # logistic이 512개 이니 그만큼 bias도 필요
# layer2 = tf.sigmoid(tf.matmul(X,W2) + b2) sigmoid에서 relu로 변경
layer2 = tf.nn.relu(tf.matmul(X,W2) + b2)



# hidden layer2
#W3 = tf.Variable(tf.random.normal([64, 32]))
W3 = tf.get_variable('W333', shape=[64, 32], 
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random.normal([32])) 
_layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
layer3 = tf.nn.dropout(_layer3, rate=0.3)

# hideen layer3
#W4 = tf.Variable(tf.random.normal([32, 16]))
W4 = tf.get_variable('W444', shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([16])) 
layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)



# output layer
#W5 = tf.Variable(tf.random.normal([16, 10])) # 최종 output은 onehot encoding의 depth
W5 = tf.get_variable('W555', shape=[16, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random.normal([10])) 

logit = tf.matmul(layer4,W5) + b5
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# 초기화
sess.run(tf.global_variables_initializer())

# epochs = 1000
for step in range(5000):
  _, loss_val = sess.run([train, loss], feed_dict={X:x_data_train_norm, T:t_data_train_onthot})

  if step % 500 ==0:
    print('loss : {}'.format(loss_val))
```



* 정확도 측정

```python
# Accuracy
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

print('정확도 : {}.'.format(sess.run(accuracy, feed_dict={X:x_data_test_norm, T:t_data_test_onthot})))
```

```
정확도 : 0.9436507821083069.
```

* 정확도가 올라간 것을 확인할 수 있었음.
* 하지만 우리가 Overfitting이 되는지 확인해봐야함.

------



## Overfitting

* Training Data에 대해 너무 적합하게 학습이 된 경우, 실제 Data 예측에서는 오히려 정확도가 떨어지는 경우
* 예방법
  1. 데이터양이 많아야 함.
  2. 필요없거나 중복이되는 feature들은 삭제
  3. L2 정규화(인위적으로 w의 값을 조정)
  4. DNN에서는 dropout이라는 방법으로 줄일 수 있음.
     1. 학습에 참여하는 각 layer안의 일정 node를 사용하지 않는 기법.
     2. 가장 효율적이고 쉬운 방법



## Dropout

* Node는 존재하지만 마치 스위치처럼 사용/비사용을 지정할 수 있음.

```python
# Dropout
model.add(Dropout(0.2))
```



## Tensorflow 2.X 버전

* 2.x 버전으로 구현해보기.

```python
# import
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# data는 위와 동일

# keras 구현
model = Sequential()
model.add(Flatten(input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2)) # 추가해도 되고 안해도 됨.
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=1e-2),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_data_train_norm,
                    t_data_train,  # loss에 sparse가 붙었기 때문에 one-hot 안해준 데이터
                    epochs=1000,
                    verbose=1,
                    validation_split=0.3,
                    batch_size=512)
```



* 정확도 측정

```python
print(model.evaluate(x_data_test_norm, t_data_test))
```

```python
accuracy: 0.9642 [0.26779064536094666, 0.964206337928772]
```

* 학습 시 history 객체의 records를 보면서 hyperparameter를 수정해나아가며 model을 튜닝해줘야함.

