# Tensorflow 2.x 로 Logistic Regression

* Titani 예제로 Logistic Regression을 이용해 Binary Classcification 수행하기.



## Import

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.optimizers import SGD, Adam
from scipy import stats
```



## Raw Data

```python
# Raw Data Loading
df = pd.read_csv('./data/titanic/train.csv')
display(df)
```



## Feature Engineering

```python
# Feature Engineering

# 필요없는 column은 삭제
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], 
              axis=1, inplace=False)

# display(df)
df['Family'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
# display(df)

sex_dict = {'male':0, 'female':1}
df['Sex'] = df['Sex'].map(sex_dict)
# display(df)

embarked_dict = {'S':0, 'C':1, 'Q':2}
df['Embarked'] = df['Embarked'].map(embarked_dict)
# display(df)
```

* 필요없는 column 먼저 삭제 및 편집해주기
* dict를 이용해서 num 데이터로 변경



## 결측치 처리

```python
# 결측치 처리
# Age,Embarked 에 존재
# Age는 median으로, Embarked는 최빈값(mode)로 처리
df.loc[df['Age'].isnull(), 'Age'] = np.nanmedian(df['Age'].values)
df.loc[df['Embarked'].isnull(), 'Embarked'] = 0
```

* loc(행,열)을 이용해서 Age column에서 column의 null값인 행을 선택.
* nanmedian을 통해 Age column의 null값 제외 중앙값을 계산.
* Embarked column의 null 값을 찾아서 0으로 대입 ( 0은 'S'값이 최빈값이여서 대체 값으로 입력)



## Category

```python
# Age categorical value로 변경해서 사용
def age_category(age):
    if (age >= 0) & (age < 25):
        return 0
    elif (age >= 25) & (age < 50):
        return 1
    else:
        return 2

df['Age'] = df['Age'].map(age_category)
display(df)
```

* category 지정하는 함수 구현
* mapping으로 함수 작동 dict와 같은 원리



## Data Split

```python
# train / test split
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(df.drop('Survived', axis=1, inplace=False),
                 df['Survived'], test_size=0.3,random_state=0)
```

* train_test_split을 사용하여 x,y에 대해 training data와 test data로 나눔.



## 정규화 처리

```python
# 정규화 처리
scaler = MinMaxScaler()
scaler.fit(x_data_train)

x_data_train_norm = scaler.transform(x_data_train)
x_data_test_norm = scaler.transform(x_data_test)
```

* MinMaxScaler를 통해 x data 정규화
* y data는 0과 1로 나누어진 Binary Classcification이기 때문에 정규화 필요 없음.



## 학습

* sklearn과 tensorflow 2.x keras로 각각 구현해보고 accurcy 비교해보기.

### Sklearn

```python
######### sklearn으로 구현
model = LogisticRegression()
model.fit(x_data_train_norm, t_data_train)
result = model.score(x_data_test_norm, t_data_test)
print('sklearn의 accuracy : {}'.format(result)) # 0.7947761194029851
```

* model 변수 안에 어떤 Regression을 쓸건지 지정.
* model 변수 안에 fit을 사용해 data 밀어넣기, 정규화된 입력변수와 종속변수 넣기
* 정규화된 test 입력변수와 test 종속변수로 score를 사용하여 평가하기



## Tensorflow

```python
######## tensorflow 2로 구현
keras_model = Sequential()

keras_model.add(Flatten(input_shape=(x_data_train_norm.shape[1],))) 
# 5, 라고 쓸 수 있지만 constant 값이 오지 않도록 shape코드로 쓰기!

keras_model.add(Dense(1, activation='sigmoid')) 
# 필요한 Logistic 개수 Binary Classcification이기 때문 1개

keras_model.compile(optimizer=SGD(learning_rate=1e-2),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

keras_model.fit(x_data_train_norm, t_data_train,
                epochs=1000, verbose=0)

keras_result = keras_model.evaluate(x_data_test_norm, t_data_test)  # kears는 score가 아닌 evaluate를 사용
print('keras의 accuracy : {}'.format(keras_result)) # 0.7947761416435242
```

* keras_model 변수 안에 Sequential() 모델 설정
* add를 이용해서 input_layer인 Flatten 넣기. 입력변수가 5개이기 때문에 5, 로 shape을 잡아줘야하지만, 그러면 constant값이 되기 때문에 정규화된 입력변수의 shape의 열을 indexing해서 넣기.
* add를 이용해서 output layer Dense 넣기. Logistic Regression 1개만 구현하기 때문에 1로 명시, activate 함수는 sigmoid 명시
* compile안에 초기화방식(SGD), learning_rate 값, loss 함수, metrics로 평가형식 지정 가능.
* fit으로 데이터 밀어넣기 verbose는 학습의 진행 상황을 보여줄 것인지 지정여부.  verbose를 1로 세팅하면 학습이 되는 모습을 볼 수 있음.
* keras는 score가 아닌 evaluate로 평가진행.