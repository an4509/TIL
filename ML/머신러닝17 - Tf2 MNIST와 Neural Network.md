# MNIST를 TF 2.x버전으로 구현

* Sklearn과 TF2로 구현해보기



## Sklearn으로 구현해보기

* Data 처리

```python
import numpy  as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings(action='ignore')

# Raw Data
df = pd.read_csv('./data/digit/train.csv')
display(df)

# 결측치와 이상치를 처리하기
# feature engineering
#  학습에 필요없는 feature는 삭제, 기존 feature들을 이용해서 새로운 column 을 생성
# binning 처리(연속적인 숫자값을 categorycal value로 변환)

# 독립변수와 종속변수를 분리
x_data = df.drop('label', axis=1, inplace=False)
t_data = df['label']  # sklearn은 one-hot 처리가 필요 없음.

# 정규화 처리
scaler  = MinMaxScaler()
scaler.fit(x_data)  # 원래는 2차원이 들어가야하는데 dataframe으로부터 numpy로 변형해줌
x_data_norm = scaler.transform(x_data)

# Data Split
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(x_data_norm, t_data, test_size=0.3, random_state=0)
```



* Sklearn 구현

```python
sklearn_model = LogisticRegression(solver='saga')
sklearn_model.fit(x_data_train, t_data_train)
print('sklearn result : ')
print(classification_report(t_data_test, sklearn_model.predict(x_data_test)))
```

```python
sklearn result : 
              precision    recall  f1-score   support

           0       0.96      0.96      0.96      1242
           1       0.95      0.97      0.96      1429
           2       0.92      0.90      0.91      1276
           3       0.91      0.90      0.91      1298
           4       0.92      0.92      0.92      1236
           5       0.88      0.88      0.88      1119
           6       0.93      0.95      0.94      1243
           7       0.94      0.93      0.94      1334
           8       0.89      0.88      0.88      1204
           9       0.89      0.89      0.89      1219

    accuracy                           0.92     12600
   macro avg       0.92      0.92      0.92     12600
weighted avg       0.92      0.92      0.92     12600
```



## 새로 배운내용 Solver

* LogisticRegression은 solver를 지정해야 함.
* 데이터가 많느냐, 적느냐에 따라 나름대로 solver를 지정해주면 그것에 맞는 model이 만들어짐.
  default로 사용되는 solver는 lbfgs라는 solver.
* 작은 데이터에 최적화 되어있는 Logistic model이 생성됨.
* 데이터량이 많아지면 performance가 좋지 않음.
* 40000개 정도되면 적은것은 아님! 그래서 다른 solver를 사용할거임.
* 많은 경우는 saga(Stochastic Average Gradientdescent/ sag의 확장판 saga)



## Tensorflow 2.x 구현해보기

* Data 처리

```python
# tensorflow 2.x 버전으로 구현해보기

# %reset
import numpy  as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD 
# keras는 optimizer가 따로 있기 때문 solver가 필요없음


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action='ignore')

# Raw Data
df = pd.read_csv('./data/digit/train.csv')
# display(df)

# 독립변수와 종속변수를 분리
x_data = df.drop('label', axis=1, inplace=False)
t_data = df['label'] # tensorflow 2.x 버전은 one-hot 처리 안해도됨.

# 정규화 처리
scaler  = MinMaxScaler()
scaler.fit(x_data)  # 원래는 2차원이 들어가야하는데 dataframe으로부터 numpy로 변형해줌
x_data_norm = scaler.transform(x_data)

# Data Split
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(x_data_norm, t_data, test_size=0.3, random_state=0)
```



* TF2로 구현해보기

```python
# Tensorflow 2.x 구현
keras_model = Sequential()  # 모델 생성
keras_model.add(Flatten(input_shape=(x_data_train.shape[1],)))
keras_model.add(Dense(10, activation='softmax'))
# layer 생성에도 순서가 있음 먼저 처리할 layer가 먼저 등장해야함
keras_model.compile(optimizer=SGD(learning_rate=1e-1),
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['sparse_categorical_accuracy'])
# onehot encoding을 사용하지 않은 경우 sparse를 앞에 붙여줘야함.
# optimizer는 속도차이가 있음.

history = keras_model.fit(x_data_train, t_data_train, epochs=100,
                          batch_size=512,
                          verbose=0,
                          validation_split=0.2)
# history는 학습을 하면서 나오는 중간에 나오는 데이터를 담아주는 객체

print(keras_model.evaluate(x_data_test, t_data_test))
```

```python
394/394 [==============================] - 0s 840us/step - loss: 0.2965 - sparse_categorical_accuracy: 0.9183
[0.29653528332710266, 0.9182539582252502]
```



### History 객체

```python
# history(변수)객체 내에 history라는 속성이 있음. dict 타입
print(type(history.history))
print(history.history.keys())
# dict_keys(['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy'])

# epochs 당 train data를 이용한 accuracy와 loss, validation data를 이용한 accuracy와 loss
# 이것을 이용하면 그래프를 그릴 수 있음.
plt.plot(history.history['sparse_categorical_accuracy'], color='r') 
plt.plot(history.history['val_sparse_categorical_accuracy'], color='b') 


plt.show() # 그래프 비교를 해보면 차이가 남.. 과대적함
# epochs를 100번을 하면 과대적합이 심해짐!
# 이것을 보고 epochs를 조절이 가능
```

![](md-images/history%20graph.PNG)

* 두 곡선 그래프가 epochs 100으로 갈 수록 차이가 벌어지는 걸 볼 수 있음.
* train data와 validation data의 격차가 벌어지는 과대적합현상이 발생.





# Machine Learning 기법

* Regression
  * Linear Regression
  * Logistic Regression
    * Binary classification
    * Multinomial classification
* KNN
* SVM(Support Vector Machine)
* Decision Tree
* Neural Network(신경망)
* 기타등등(강화학습, Naive Bayes,...)



## Neural Network(신경망)

* 최종 목표인 AI를 구현하기 위해 사람의 뇌를 연구하기 시작.

* neuron이라고 불리는 뇌 신경세포의 동작으로 `사고`라는 것이 일어남.

* 1960년에 이러한 원리로 로랜블랫의 perceptron 탄생.

  * 다수의 신호를 입력 받아서 하나의 신호를 출력.
  * Single-Layer Perceptron Network
    * Logistic 과 비슷함.

*  Single Layer Perceptron을 활용해서 computer 회로를 만들어내고자함.

  * 여러 gate로 구성

    * and, or, xor, not ,nand을 구현
    * 코드로 and, or, xor이 구현되는지 알아보기~!

    ```python
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import classification_report
    
    # Training Data Set
    x_data = np.array([[0,0],
                       [0,1],
                       [1,0],
                       [1,1]], dtype=np.float32)
    # # And t_data
    # t_data = np.array([[0], [0], [0], [1]], dtype=np.float32)
    
    # # OR t_data
    # t_data = np.array([[0], [1], [1], [1]], dtype=np.float32)
    
    # XOR t_data
    t_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # placeholder
    X = tf.placeholder(shape=[None,2], dtype=tf.float32)
    T = tf.placeholder(shape=[None,1], dtype=tf.float32)
    
    # Weight & Bias
    W = tf.Variable(tf.random.normal([2,1]))
    b = tf.Variable(tf.random.normal([1]))
    
    # Hypothesis
    logit = tf.matmul(X,W) + b
    H = tf.sigmoid(logit)
    
    # loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,
                                                                  labels=T))
    
    # train 
    train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)
    
    # Session & 초기화
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for step in range(30000):
        _, loss_val = sess.run([train, loss],feed_dict={X:x_data, T:t_data})
        
        if step % 3000 == 0:
            print('loss : {}'.format(loss_val))
    ```

    ```python
    # Accuracy
    accuracy = tf.cast(H>=0.5, dtype=tf.float32)
    result = sess.run(accuracy, feed_dict={X:x_data})
    print(classification_report(t_data.ravel(), result.ravel()))
    # and gate 는 제대로 동작하는 것을 확인했음!
    # OR gate도 제대로 동작
    # XOR가 학습이 제대로 되지 않는 것을 확인 함.
    # 이유는 구분선을 못그령 ㅜㅜ
    # 이게 되야
    ```

    ```python
         precision    recall  f1-score   support
    
             0.0       0.33      0.50      0.40         2
             1.0       0.00      0.00      0.00         2
    
        accuracy                           0.25         4
       macro avg       0.17      0.25      0.20         4
    weighted avg       0.17      0.25      0.20         4
    ```

    * Logistic Regression으로 and와 or 을 구현해서 prediction이 잘 나오는지 해보기
    * neuron과 동일한 동작을 하는 perceptron이 gate 구현을 할 수 있는지 실습.
    * XOR이 안되니까 어떻게 하면 Perceptron으로 학습시킬 수 있는지 연구
    * 민스키 교수가 Multi Layer Perceptron으로 해야 가능하지만 학습이 어려움.
    * 이후 침체기가 옴.





