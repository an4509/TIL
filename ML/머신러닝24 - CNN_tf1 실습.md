# CNN 수행평가1 오답노트



## Multinomial Classification

* 이미지 확인 과정 생략함.  데이터 전처리 전에 눈으로 확인하는 것이 좋음!

```python
# # 이미지 확인
img_data = df.drop('label', axis=1, inplace=False).values
print(img_data.shape) # (60000, 784)

fig = plt.figure()
fig_arr = list()

for n in range(10):
    fig_arr.append(fig.add_subplot(2,5,n+1))
    fig_arr[n].imshow(img_data[n].reshape(28,28), cmap='gray')

plt.tight_layout()
plt.show()
```

* list를 만들어서 for문으로 subplot 10개 만들고 img_data를 10개 슬라이싱하여  reshape과 동시에 넣어주기.
* 그 외에는 차이 없음.



## DNN

* batch 안 넣어줌.

```python
num_of_epoch = 1000
batch_size = 100

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 반복학습
for step in range(num_of_epoch):
    
    total_batch = int(x_data_train_norm.shape[0] / batch_size)
    
    for i in range(total_batch):
        batch_x = x_data_train_norm[i*batch_size:(i+1)*batch_size]
        batch_t = t_data_train_onehot[i*batch_size:(i+1)*batch_size]
        _, loss_val = sess.run([train, loss], feed_dict={X:batch_x,
                                                         T:batch_t})
    if step % 100 == 0:
        print('Loss : {}'.format(loss_val))
```

* 이중 for문으로 batch size 정한 값으로 data 나눠서 학습시켜주기
* batch를 넣어주니까 정확도가 올라갔음 / 0.90



# CNN

* input layer 위치를 잘 못 알고 있었음.
* 무조건 same이라고 해서 원본과 결과이미지 크기가 같지는 않음!
* Pooling layer에서 dense layer로 넘어갈 때 4차원을 2차원으로 변형해줘야함. 이때,  (?, 7, 7, 64)에서 7,7,64 를 곱해주어 열로 reshape 해야함.
* 또한 차원 변형 때 np. array가 아닌 tensor를 이용해서 차원변형이 가능함.
* conv layer도 np가 아닌 tensorflow에 결합해야 하기 때문에 tf.Variable에 넣어줘야함.

```python
# CNN tensorflow 1.15

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report


# raw data
df = pd.read_csv('/content/drive/MyDrive/Machine Learning Colab/fashion-mnist_train.csv')
print('raw data shape : {}'.format(df.shape)) # (60000, 785)

# # 이미지 확인
img_data = df.drop('label', axis=1, inplace=False).values
print(img_data.shape) # (60000, 784)

fig = plt.figure()
fig_arr = list()

for n in range(10):
    fig_arr.append(fig.add_subplot(2,5,n+1))
    fig_arr[n].imshow(img_data[n].reshape(28,28), cmap='gray')

plt.tight_layout()
plt.show()

# train, test 데이터 분리
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(img_data, df['label'], test_size=0.3, random_state=0)

# Normalization
scaler = MinMaxScaler()
scaler.fit(x_data_train)
x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)

# One-hot encoding
sess = tf.Session()
t_data_train_onehot = sess.run(tf.one_hot(t_data_train, depth=10))
t_data_test_onehot = sess.run(tf.one_hot(t_data_test, depth=10))

# Input layer
X = tf.placeholder(shape=[None,784], dtype=tf.float32)
T = tf.placeholder(shape=[None,10], dtype=tf.float32)


## convolution ##
# x_data 4차원으로 변환
x_img = tf.reshape(X, [-1,28,28,1])
print(x_img.shape)   # (?, 28, 28, 1)

# conv layer1
W1 = tf.Variable(tf.random.normal([3,3,1,32]))
L1 = tf.nn.conv2d(x_img,
                  W1,
                  strides=[1,1,1,1],
                  padding='SAME')
L1 = tf.nn.relu(L1)
print(L1.shape)   # (?, 28, 28, 32)

# pooling layer1
# kernel size = stride = 2
P1 = tf.nn.max_pool(L1,
                    ksize=[1,2,2,1],
                    strides=[1,2,2,1],
                    padding='SAME')
print(P1.shape)   # (?, 14, 14, 32)

# conv layer2
W2 = tf.Variable(tf.random.normal([3,3,32,64]))
L2 = tf.nn.conv2d(P1,
                  W2,
                  strides=[1,1,1,1],
                  padding='SAME')
L2 = tf.nn.relu(L2)
print(L2.shape)   # (?, 14, 14, 64)

# pooling layer2
P2 = tf.nn.max_pool(L2,
                    ksize=[1,2,2,1],
                    strides=[1,2,2,1],
                    padding='SAME')
print(P2.shape)   # (?, 7, 7, 64)

# 2차원으로 변환
P2 = tf.reshape(P2, [-1,7*7*64])

# hidden layer1
W3 = tf.get_variable('W3', shape=[7*7*64, 256], initializer=tf.contrib.layers.variance_scaling_initializer()) 
b3 = tf.Variable(tf.random.normal([256]))
layer3 =  tf.nn.relu(tf.matmul(P2,W3) + b3)

# hidden layer2
W4 = tf.get_variable('W4', shape=[256, 128], initializer=tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.random.normal([128])) 
layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)
# layer3 = tf.nn.dropout(_layer3, rate=0.3)

# output layer
W5 = tf.get_variable('W5', shape=[128, 10], initializer=tf.contrib.layers.variance_scaling_initializer())
b5 = tf.Variable(tf.random.normal([10])) 

logit = tf.matmul(layer4,W5) + b5
H = tf.nn.softmax(logit)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

# epochs & batch_size
num_of_epoch = 1000
batch_size = 100

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 반복학습
for step in range(num_of_epoch):
    
    total_batch = int(x_data_train_norm.shape[0] / batch_size)
    
    for i in range(total_batch):
        batch_x = x_data_train_norm[i*batch_size:(i+1)*batch_size]
        batch_t = t_data_train_onehot[i*batch_size:(i+1)*batch_size]
        _, loss_val = sess.run([train, loss], feed_dict={X:batch_x,
                                                         T:batch_t})
    if step % 100 == 0:
        print('Loss : {}'.format(loss_val))

# Accuracy
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

print('CNN 정확도 : {}.'.format(sess.run(accuracy, feed_dict={X:x_data_test_norm, T:t_data_test_onehot})))

# test.csv 
final_df = pd.read_csv('/content/drive/MyDrive/Machine Learning Colab/fashion-mnist_test.csv')

final_x_data = final_df.drop('label', axis=1, inplace=False).values
final_t_data = final_df['label'].values

final_scaler = MinMaxScaler()
final_scaler.fit(final_x_data)
final_x_data_norm = final_scaler.transform(final_x_data)

print(classification_report(final_t_data,
                           sess.run(predict, feed_dict={X:final_x_data_norm})))
```

```python
CNN 정확도 : 0.9029444456100464.
              precision    recall  f1-score   support

           0       0.86      0.86      0.86      1000
           1       0.98      0.98      0.98      1000
           2       0.86      0.86      0.86      1000
           3       0.91      0.91      0.91      1000
           4       0.84      0.87      0.86      1000
           5       0.98      0.96      0.97      1000
           6       0.77      0.74      0.76      1000
           7       0.95      0.95      0.95      1000
           8       0.98      0.97      0.98      1000
           9       0.95      0.97      0.96      1000

    accuracy                           0.91     10000
   macro avg       0.91      0.91      0.91     10000
weighted avg       0.91      0.91      0.91     10000
```



