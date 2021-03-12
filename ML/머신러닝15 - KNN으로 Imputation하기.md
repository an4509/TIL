# KNN으로 Imputation하기

* 입력값 x에 대해 하는 것이 아니라 종속변수(t)에 대해 보간하는 것임.
* 독립변수에도 결측치가 있을 때 평균화 기법을 주로 사용.



## Ozone Data 보간 Process

1. 독립변수를 먼저 보간한 다음, 예측기법으로 종속변수를 채워줄 예정.
2. 독립변수(solar, wind, Temp)에 대한 결측치를 찾아서 median으로 처리.
   * 이상치처리가 안되서 평균에 영향을 줄 수 있으므로 median 사용.
3. 독립변수에 대한 이상치를 찾아서 mean으로 처리.
4. 정규화 진행.
5. KNN을 이용한 종속변수에 대한 결측치 처리.



## 내가 구현한 코드

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats # 이상치처리
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor # regressor와 clossifier 구분하기


df = pd.read_csv('./data/ozone.csv')
df.drop(['Month','Day'], axis=1, inplace=True)
df.rename(columns = {"Solar.R": "Solar"}, inplace=True)
# display(df) # 153 rows × 4 columns

# 결측치 확인
# df.Wind.isnull().sum() # 0
# df.Temp.isnull().sum() # 0
# df.Solar.isnull().sum() # 7

# s_data = df['Solar']
# display(s_data)
# s_data = s_data.dropna(how='any')
# display(s_data)
# solar_medi = np.median(s_data)
# print(solar_medi) # 205.0

# 결측치 채워주기
df['Solar'].fillna(205.0,inplace=True)
# display(df)


# 이상치 처리
# fig = plt.figure()  
# fig_1 = fig.add_subplot(1,3,1)  
# fig_2 = fig.add_subplot(1,3,2)
# fig_3 = fig.add_subplot(1,3,3)

# fig_1.boxplot(df['Solar'])
# fig_2.boxplot(df['Wind']) # wind에만 이상치 3개 존재
# fig_3.boxplot(df['Temp'])

# fig.tight_layout()
# plt.show()

# wind mean 확인
# wind_mean = df.Wind.mean()
# print(wind_mean) # 10.0

df.loc[(df['Wind']>=17.5), 'Wind']=10.0
# display(df)
# print(df['Wind'])

# fig = plt.figure()  
# fig_1 = fig.add_subplot(1,1,1)
# fig_1.boxplot(df['Wind'])
# fig.tight_layout()
# plt.show()

# data set
train_x_data = df[['Solar', 'Wind','Temp']].values
train_t_data = df['Ozone']

# 이상치 처리
# transform 안에 2차원이 들어가야함
scaler = MinMaxScaler()
scaler.fit(train_x_data)
x_data_train_norm = scaler.transform(train_x_data)

# KNN으로 종속변수 결측치 처리
```



## 오답노트

* Raw Data에서 사용하지 않는 column을 처음부터 제외하고 x,t 독립변수를 잡아주기.
* 각 3개의 column의 결측치와 이상치를 확인하는 과정에서 반복될 때 for문으로 쓰기.
* 이상치를 처리할 때 z-score로 처리하지 않았음.
* 종속변수의 이상치 확인을 하지 않음.
* 시간이 부족하여 KNN으로 Imputation을 하지 못함.



## 새로 알게된 함수

```python
np.nanmedian()
# nan을 제외하고 median 값을 계산.
```



## 강의코드

```python
# Ozone data를 이용한 Tensorflow 2.x Linear Regression 구현

# Data 전처리
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import warnings

warnings.filterwarnings(action='ignore')   # warning 출력을 하지 않아요!

# Raw Data Loading
df = pd.read_csv('./ozone.csv')
# display(df.head(), df.shape)

# 독립변수 & 종속변수
x_data = df[['Solar.R', 'Wind', 'Temp']]  # Fancy indexing
t_data = df['Ozone']

# 1. 독립변수에 대한 결측치를 검출한 후 Imputation을 진행(평균화기법-median)
#    median으로 처리하는 이유는 이상치를 처리하지 않았기 때문.
for col in x_data.columns:
    col_median = np.nanmedian(x_data[col])
    x_data[col].loc[x_data[col].isnull()] = col_median
    
# 2. 독립변수에 대한 이상치를 검출한 후 mean값으로 처리할께요!
zscore_threshold = 1.8   # z-score outlier 임계값으로 사용

for col in x_data.columns:
    outlier = x_data[col][(np.abs(stats.zscore(x_data[col])) > zscore_threshold)]
    col_mean = np.mean(x_data.loc[~x_data[col].isin(outlier),col])
    x_data.loc[x_data[col].isin(outlier),col] = col_mean
    
# 3. 종속변수에 대한 이상치를 검출 한 후 mean값으로 처리할께요!     
outlier = t_data[(np.abs(stats.zscore(t_data)) > zscore_threshold)]
col_mean = np.mean(t_data[~t_data.isin(outlier)])
t_data[t_data.isin(outlier)] = col_mean

# 4. 정규화
scaler_x = MinMaxScaler()
scaler_t = MinMaxScaler()

scaler_x.fit(x_data.values)
scaler_t.fit(t_data.values.reshape(-1,1))

x_data_norm = scaler_x.transform(x_data.values)
t_data_norm = scaler_t.transform(t_data.values.reshape(-1,1)).ravel()

# 5. 종속변수에 대한 결측치를 KNN을 이용하여 Imputation 처리
# KNN 학습에 사용될 x_data와 t_data를 추려내야 해요!
x_data_train_norm = x_data_norm[~np.isnan(t_data_norm)]
t_data_train_norm = t_data_norm[~np.isnan(t_data_norm)]

knn_regressor = KNeighborsRegressor(n_neighbors=2)
knn_regressor.fit(x_data_train_norm,t_data_train_norm)

knn_predict = knn_regressor.predict(x_data_norm[np.isnan(t_data_norm)])
t_data_norm[np.isnan(t_data_norm)] = knn_predict

## 최종적으로 얻은 데이터
## 독립변수 :  x_data_norm
## 종속변수 : t_data_norm

## 이 데이터를 이용해서 sklearn과 tensorflow 2.x로 구현을 진행
```



