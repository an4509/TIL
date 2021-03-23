# Pooling Layer

**개념**

Filter를 여러개 사용하기 때문에 전체적인 데이터량이 상당히 커지는 것을 줄이기 위해 사용하는 기법.

주로 Max Pooling 기법을 사용함.



**특징**

Pooling 처리를 하려면 2가지 설정

* kernel size

* stride

보통 2가지를 같게 설정함.



* 코드로 구현하기

```python
import numpy as np
import tensorflow as tf

# pooling layer
# 입력이미지(Feature Map) / 원래는 relu 처리하여 activation map이 들어오는데 생략.
# (이미지 개수, 이미지 height, 이미지 width, channel)
# (1, 4, 4, 1)
image = np.array([[[[13],[20],[30],[0]],
                   [[8],[12],[3],[0]],
                   [[34],[70],[33],[5]],
                   [[111],[80],[10],[23]]]], dtype=np.float32)

print(image.shape) # (1, 4, 4, 1)

# ksize = 2
# stride = 2
pooling = tf.nn.max_pool(image,
                         ksize=[1,2,2,1],  # 맨 끝은 더미, 가운데 2개가 2x2를 뜻함.
                         strides=[1,2,2,1], # 통상적으로 1을 씀.
                         padding='VALID')

sess = tf.Session()
result = sess.run(pooling)
print('Pooling한 결과 : \n{}'.format(result))
print(result.shape) # (1, 2, 2, 1)
```

```
(1, 4, 4, 1)
Pooling한 결과 : 
[[[[ 20.]
   [ 30.]]

  [[111.]
   [ 33.]]]]
(1, 2, 2, 1)
```





* conv 처리와 pooling 처리하기 ( relu 제외)

```python
# Gray-scale 이미지를 이용해서 Convolution처리와 Pooling처리해보기.
# %reset

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img

fig = plt.figure()
fig_1 = fig.add_subplot(1,3,1)
fig_2 = fig.add_subplot(1,3,2)
fig_3 = fig.add_subplot(1,3,3)

ori_image = img.imread('./data/image/girl-teddy.jpg')
fig_1.imshow(ori_image)
# print(ori_image.shape) (429, 640, 3) # 3차원 데이터

# convolution 처리

# 처리하기 위해 4차원으로 변환하기 (이미지개수, height, width, color(channel))
# 만약 이미지의 개수가 다수이면 모든 이미지가 동일한 크기여야함.
input_image = ori_image.reshape((1,) + ori_image.shape)
print(input_image.shape) # (1, 429, 640, 3)

# channel 1로 줄이기
channel_1_input_image = input_image[:,:,:,0:1]
channel_1_input_image = channel_1_input_image.astype(np.float32)
print(channel_1_input_image.shape) # (1, 429, 640, 1)

# filter (filter height, filter width, filter channel, filter의 개수)
# (3,3,1,1)
weight = np.array([[[[-1]],[[0]],[[1]]],
                   [[[-1]],[[0]],[[1]]],
                   [[[-1]],[[0]],[[1]]]], dtype=np.float32)
# stride : 1
# padding : VALID
sess = tf. Session()
conv2d = tf.nn.conv2d(channel_1_input_image,
                     weight,
                     strides=[1,1,1,1],
                     padding='VALID')

conv2d_result = sess.run(conv2d)
print(conv2d_result.shape) # (1, 427, 638, 1)

# 이미지 표현을 위해 3차원으로 변환하기
t_img =  conv2d_result[0,:,:,:]
fig_2.imshow(t_img)


# pooling 처리

# ksize = 3
# stride = 3
pooling = tf.nn.max_pool(conv2d_result,
                         ksize=[1,3,3,1],
                         strides=[1,3,3,1],
                         padding='VALID')

pooling_result = sess.run(pooling)
print(pooling_result.shape) # (1, 142, 212, 1)

# 이미지 표현을 위해 3차원으로 변환하기
p_image = pooling_result[0,:,:,:]
fig_3.imshow(p_image)

plt.show()
```

![](md-images/pooling%20image.PNG)



* MNIST data로 CNN 전체 구현해보기.

```python
# %reset

# filter 설정을 다양하게 해보기

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img

fig = plt.figure()
fig_list = list() # 안에 각각의 subplot을 저장할 것임.

for i in range(5):
    fig_list.append(fig.add_subplot(1,5,i+1))
    
# Raw Data Loading
df  = pd.read_csv('./data/digit/train.csv')

img_data = df.drop('label', axis=1, inplace=False).values
print(img_data.shape)

# 샘플로 사용할 이미지를 하나 선택
ori_image = img_data[5:6].reshape(28,28)
fig_list[0].imshow(ori_image, cmap='gray')

# convolution 처리
# 4차원으로 변환하기 (1, 28, 28, 1)로 바꾸기
print(ori_image.shape) # (28, 28)
input_image = ori_image.reshape((1,) + ori_image.shape + (1,))
print(input_image.shape) # (1, 28, 28, 1)

# 실수로 바꿔주기
input_image = input_image.astype(np.float32)

# filter(kernel)
# (3,3,1,4) 3x3 channel1개, filter 4개
weight = np.random.rand(3,3,1,4) # 형태 안에 난수값들이 들어감.
print(weight.shape) # (3, 3, 1, 4)

# stride : 1
# padding : VALID

sess = tf.Session()

conv2d = tf.nn.conv2d(input_image,
                      weight,
                      strides=[1,1,1,1],
                      padding='VALID')

conv2d_result = sess.run(conv2d)

# ReLU  (Rectified Linear Unit) 처리
relu_ = tf.nn.relu(conv2d_result)
relu_result = sess.run(relu_)


# pooling 처리
# kernel size = stride = 2
pooling =  tf.nn.max_pool(relu_result,
                          ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='VALID')

pooling_result = sess.run(pooling)

print(pooling_result.shape) # (1, 13, 13, 4)
# (4, 13, 13, 1)로 바꿔주기

# 축 교환해주기
i = np.swapaxes(pooling_result,0,3)
print(i.shape) # (4, 13, 13, 1)

for filter_idx, t_img in enumerate(i): # enumerate는 맨 앞을 index로 잡아 기준으로 뒤에 데이터를 뽑아줌.
    fig_list[filter_idx+1].imshow(t_img, cmap='gray')
```

![](md-images/pooling%20image2.PNG)



# CNN 수행평가 01

Multinomial Classification, DNN, CNN 각각의 모델을 만들어서 Accuracy 비교하기



* Multinomial Classification (코드가 이전과 중복되므로 생략)

```python
우리 Model의 최종 정확도는 : 0.846666693687439
```



* DNN 

```python
# 이전 생략

# input layer
X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
T = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# hidden layer1
W2 = tf.get_variable('W2', shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random.normal([512]))
layer2 = tf.nn.relu(tf.matmul(X,W2) + b2)

# hidden layer2
W3 = tf.get_variable('W3', shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random.normal([256]))
layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)

# hidden layer3
W4 = tf.get_variable('W4', shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([128]))
layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)

# output layer
W5 = tf.get_variable('W5', shape=[128, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random.normal([10])) 

# 이후 생략

정확도 : 0.862500011920929.
```



* CNN

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# raw data
df = pd.read_csv('/content/drive/MyDrive/Machine Learning Colab/fashion-mnist_train.csv')
print('raw data shape : {}'.format(df.shape)) # (60000, 785)

# # 이미지 확인
img_data = df.drop('label', axis=1, inplace=False).values
print(img_data.shape) # (60000, 784)
# plt.imshow(img_data[0].reshape(28,28), cmap='gray')
# plt.show()

x_data = img_data.reshape(-1, 28, 28, 1).astype(np.float32)
print('x_data shape: {}'.format(x_data.shape))



## convolution ##
# filter(kernel)
weight = np.random.rand(3,3,1,4) # 형태 안에 난수값들이 들어감.
print('filter shape : {}'.format(weight.shape)) # (3, 3, 1, 4)

# stride : 1
# padding : VALID

sess = tf.Session()

conv2d = tf.nn.conv2d(x_data,
                      weight,
                      strides=[1,1,1,1],
                      padding='VALID')

conv2d_result = sess.run(conv2d)

# ReLU  (Rectified Linear Unit) 처리
relu_ = tf.nn.relu(conv2d_result)
relu_result = sess.run(relu_)

# pooling 처리
# kernel size = stride = 2
pooling =  tf.nn.max_pool(relu_result,
                          ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME')

pooling_result = sess.run(pooling)
print('pooling_result shape: {}'.format(pooling_result.shape))
# pooling_result shape: (60000, 13, 13, 4)


# train, test 데이터 분리
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(pooling_result, df['label'], test_size=0.3, random_state=0)

# 차원변환 여기가 어렵다 어떻게 변환해야하지?
# x_data_train = x_data_train.reshape(-1,-1)
# x_data_test = x_data_test.reshape(-1,-1)

# print(x_data_train.shape)
```

문제 

* CNN에서 DNN 넘어갈 때 차원이 설정을 어떻게 해야되는지 모르겠음.







