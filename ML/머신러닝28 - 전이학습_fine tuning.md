# Pretrained Network

* 저번시간에 Pretrainde Network를 이용해서 빠르고 accuracy를 향상시킨 모델을 만들었음. 
* 정확도를 더 올릴려면 데이터 증식을 해야함.
* 데이터 증식과 CNN 모델에 합쳐서 코드 구현해보기



**Import 및 경로설정**

```python
# 조금 더 나은 결과를 얻으려면
# 데이터를 증식시켜야 함.
# Pretrained Network과 classifier를 합쳐서 모델을 만들것임.

import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

base_dir = '/content/drive/MyDrive/Machine Learning Colab/CAT_DOG/cat_dog_full'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
```



**ImageDataGenerator**

```python
train_datgen = ImageDataGenerator(rescale=1/256,
                                  rotation_range=40,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True)

validaton_datgen = ImageDataGenerator(rescale=1/256)
```

* `train data`는 데이터 증식을 위해 여러 설정을 하지만, `validation data`는 검증만을 하는 데이터이기 때문에 데이터증식 설정을 해줄 필요가 없음.



**flow_from**

```python
train_generator = train_datgen.flow_from_directory(
    train_dir,
    classes=['cats', 'dogs'], # 명시하지 않으면 순서대로 0부터 매겨짐
    target_size = (150,150),
    batch_size = 100,
    class_mode = 'binary'
)

validation_generator = validaton_datgen.flow_from_directory(
    validation_dir,
    classes=['cats', 'dogs'], # 명시하지 않으면 순서대로 0부터 매겨짐
    target_size = (150,150),
    batch_size = 100,
    class_mode = 'binary'
)
```

* `train dir` : 어떤 경로의 directory를 사용할 건지 명시
* `classes` : 0~9 사이 숫자로 명시 순서대로 class 값 설정.
* `target_size` : 데이터 generate 크기
* `batch_size` : 한번에 가져올 이미지 개수
* `class_mode` : 분류 유형 ('binary' or 'categorical')



 **Pretrained Network을 삽입한 모델 구현**

```python
# Pretrained Newtwork
model_base = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(150,150,3))

model_base.trainable=False # pretrained network의 param을 학습시키지 않고 동결시키기 위한 코드

model = Sequential()

model.add(model_base)

model.add(Flatten(input_shape=(4*4*512,))) 

model.add(Dense(units=256,
                activation='relu',))
model.add(Dropout(0.6))
model.add(Dense(units=1,
                activation='sigmoid'))

model.summary()

model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=140,
                    epochs=30,
                    validation_data = validation_generator,
                    validation_steps=60)
```

* Pretrained network를 삽입한 모델을 구현할 때 주의할 점은 network들의 parameter들이 함께 학습되지 않게 동결시켜줘야함
* 동결시켜주는 코드 `model_base.trainable=False`를 넣어줘야함.



# Fine Tuning

* 번역으로는 미세조정.

* fine tuning은 pretrainde network의 parameter를 모두 동결시키지 않고
* 상위에 있는 몇개의 convolution layer를 동결 해제해서 param을 같이 학습



## 절차

1. base netwokr(Pretrainde network) 위에 새로운 network(FC Layer)을 추가
2. base network을 동결
3. 새로 추가된 FC Layer을 학습
4. base network의 일부분 layer를 동결에서 해제
5. 동결 해제한 layer와 FC Layer를 다시 학습



## 코드구현

```python
### 이전 코드는 위와 동일
# 동결 해제
model_base.trainable  = True

for layer in model_base.layers:
    if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv1']:
        layer.trainable = True
    else:
        layer.trainable = False

# 일반적으로 learning_rate를 더 작게 설정
model.compile(optimizer=RMSprop(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=140,
                    epochs=30,
                    validation_data = validation_generator,
                    validation_steps=60)
```

* `model_base.trainable  = True` 로 동결을 전체 해제
* `for` 문과 `if` 문으로 동결해제할 layer 이름을 명시하여 동결해제 외에는 다시 False
* `learning_rate` 는 일부 동결해제한 후 재학습할 때 이전보다 더 작게 잡아줘야함.



