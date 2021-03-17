# Neuron

* 이전 neuron으로 부터  입력신호를 전달받아서 또 다른 신호를 발생시키는 일을 함.
* 입력값에 비례해서 출력값을 내보내는 형태가 아님.
* 입력값에 가중치(w)를 곱한 후 여러가지 값(b)을 모두 더해서 특정 함수(Activation Function)를 이용해, threshold를 넘는지 확인 임계점을 넘으면 출력값 발생



## Neural Network

* 실제 신경망을 따서 인공 신경망을 구성
* Logistic Regression가 하나의 neuron처럼 역할을 하며 각각 연결되어 있음.
* Logistic Regression이 모여 있는 층을 layer라 지칭.



## Deep Network

* Deep Learning
* 1개의 Logistic Regression을 나타내는 node가 서로 연결되어 있는 신경망 구조를 바탕으로 Input Layer(입력층) + 1개 이상의 은닉층 + Ouput Later(출력층)을 구조.
* 출력층의 오차를 기반으로 각 node의 가중치를 학습시키는 Machine Learning의 한 분야
* 1개 이상의 hidden layer를 이용하면 model의 정확도(accuracy)가 높아짐.
* hidden layer를 깊게(여러개 사용) 할 수록 정확도가 더 높아짐.



1. 장점
   * 정확도가 가장 높다.
2. 단점
   * 학습하는 시간이 오래 걸림.



# GPU 사용방식

1. 그래픽카드 GPU가 달려있는 경우 (NVIDIA)
2. Colab
3. AWS , Google Cloud



# Deep Learning

* n이 늘어나면 3 이상을 넘어가면 성능이 dramatic하게 증가하지는 않음.
* n이 클수록 정교한 학습이 증가하긴 하지만 정확도가 증가하는 폭이 점점 작아짐.
* 이전 Layer의 출력의 개수와 이후 Layer의 입력의 개수가 동일해야함. (Fully Connected Network, Dense layer)



## Input Layer

* 데이터를 전달하는 역할
* 데이터를 다음 node로 전달할 때 사이에 가중치(W) 값이 있음.



## Hidden Layer

* 입력값에 가중치가 곱해져서 전달된 데이터는  node에 있던 bias가 더해지고 summation (Linear Regression).
* 이후 sigmoid로 activation하면 Logistic 되어 다음 node로 전달. 



## Output Layer

* output layer에 있던 node가 1개면 sigmoid 함수를, 여러개면 softmax 처리하여 출력.



## Propagation

* Data의 흐름
* feed forward



## DNN 코드구현

* Single Layer Perceptron XOR 코드 구현이 안되는 문제 해결해보기.
* input layer 1개 / node 2개(입력 parameter의 개수)
* hidden layer 2개 / node1 10개, node2 6개 (우리가 정해주는 것)
* output layer 1개 / node 1개(Logistic 의 개수)
* tf 1.15와 tf 2.x 버전으로 예제 실행

```python
# tensorflow 1.15 버전으로 XOR 문제를 학습하기

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

#  Training Data Set
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
t_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# placeholder
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
T = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & bias
W2 = tf.Variable(tf.random.normal([2,10]))
# hidden layer 1번째 node가 10개이므로 입력이 10가 들어가야함 그러므로 W 출력이 10이 나와야 입력으로 들어갈 수 있음.
b2 = tf.Variable(tf.random.normal([10]))
layer2 = tf.sigmoid(tf.matmul(X,W2) + b2) # 다음 hidden layer에 input으로 들어감.

# hidden layer2
W3 = tf.Variable(tf.random.normal([10,6]))
b3 = tf.Variable(tf.random.normal([6]))
layer3 = tf.sigmoid(tf.matmul(layer2,W3) + b3) 

# output layer
W4 = tf.Variable(tf.random.normal([6,1])) # 입력데이터 6, 0과 1를 표현하는 binary기 때문에 logistic 1개만 필요하기 때문
b4 = tf.Variable(tf.random.normal([1]))

# hypothesis
logit = tf.matmul(layer3,W4) + b4
H = tf.sigmoid(logit)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=T))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(30000):
  _, loss_val = sess.run([train, loss], feed_dict={X:x_data, T:t_data})

  if step % 3000 == 0:
    print('loss : {}'.format(loss_val))
    
accuracy = tf.cast(H>=0.5, dtype=tf.float32)
result = sess.run(accuracy, feed_dict={X:x_data})

print(classification_report(t_data.ravel(), result.ravel()))
```





```python
# Tensorflow 2.x 버전으로 구현해보기

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report

#  Training Data Set
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
t_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# keras 구현
model = Sequential()

# layer를 추가
model.add(Flatten(input_shape=(2,)))      # input layer
model.add(Dense(10,activation='relu')) # hidden layer # 원래는 sigmoid지만 잘 안되니 relu로 바꿔서 실행
model.add(Dense(30,activation='relu'))  # hidden layer # deep 할수록 학습이 더 잘 됨.
model.add(Dense(20,activation='relu'))  # hidden layer # node가 많을 수록 학습이 더 잘됨.
model.add(Dense(6,activation='relu'))  # hidden layer
model.add(Dense(1,activation='relu'))  # output layer

# compile
model.compile(optimizer=SGD(learning_rate=1e-2), loss='binary_crossentropy', metrics=['accuracy'])

# 학습
history = model.fit(x_data, t_data, epochs=1000, verbose=0)

predict_val = model.predict(x_data)
result = tf.cast(predict_val >= 0.5, dtype=tf.float32).numpy().ravel()
# tf2 라서 session이 존재하지 않기 때문에 node를 numpy를 통해 값을 빼오기
print(classification_report(t_data.ravel(),result))

```





