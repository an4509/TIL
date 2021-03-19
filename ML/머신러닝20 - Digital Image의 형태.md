# Digital Image의 형태

**이미지 좌표계**

* 우리가 일반적으로아는 데카르트 좌표계와 다름
* Matrix 구조
* ndarray로 표현하면 (행, 열)
* 행은 y축 열은 x축
* 순서가 (y,x)



**Binary Image (이진 이미지)**

* 각 pixel의 값을 0 or 1로 표현 
* Grey-scale과 다름. (8bit 전부 사용)
* 각 pixel이 1bit만 있으면 표현 가능
* 실제로는 1개의 pixel은 8bit를 사용. 
* 그래서 8bit이 공간이 잡히고 1개 bit 사용되고 7bit는 공간이 남음.



**Grey-scale Image**

* 흑백 이미지

* 각 pixel의 값을 0~255의 값으로 표현
* 1pixel에 8bit를 사용 2^8 = 256개



**color Image**

* 3개의 channel이 포함 ( 각 픽셀의 값이 3개의 값이 있는 것을 지칭 / RGB)
* 24 bit를 이용 (True color)
* png는 4채널이므로 32bit (RGBA)
* 현업에서는 grey-scale로 변환해서 모양을 학습시킴.
* color image를 grey-scale 변환 알아보기.



## 이미지 처리하기

1. 이미지 불러오기

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Image Open => Image 객체로 생성
img = Image.open('./data/image/justice.jpg')

print(type(img)) # <class 'PIL.JpegImagePlugin.JpegImageFile'>

plt.imshow(img)
plt.show() # 이미지 보이기
```



2. pixel값 확인하기

```python
pixel = np.array(img)
print('x좌표 : {}, y좌표 : {}의 pixel값은 : {}'.format(100,200,pixel[200,100]))
# x좌표 : 100, y좌표 : 200의 pixel값은 : [120  83  54]
# 한 픽셀의 값이 3개가 있는 것을 알 수 있음. 3차원 데이터

print('이미지의 크기 : {}'.format(img.size)) # 이미지의 크기 : (640, 426)
print('이미지의 shape : {}'.format(pixel.shape)) # 이미지의 shape : (426, 640, 3)
# png는 행, 열, 4 형식 투명도까지 포함되어 있음.
```



3. 여러처리

```python
# 이미지에 대한 기본 처리
# 이미지 객체를 이용해서 여러처리를 해보기.

# 이미지 저장
img.save('./data/image/my_image.jpg') # 이미지 저장

# 이미지 잘라내기 (좌상, 우하)
crop_img = img.crop((30,100,150,330)) # 튜플로 표현하기
plt.imshow(crop_img)
plt.show()

# 이미지 크기 변경
resize_img = img.resize((int(img.size[0]/8),int(img.size[1]/8))) # 튜플로 받음
plt.imshow(resize_img)
plt.show()

# 이미지 회전
rotate_img = img.rotate(180)
plt.imshow(rotate_img)
plt.show()
```



## 컬러 이미지를 흑백으로 변환하기

* 평균을 내는 방법
* L = 0.2999R + 0.5870G + 0.114B (opencv 방식)
* 휘도를 이용하는 방법
* 명도, 채도를 빼는 방법



### 평균내는 방법으로 변환하기

```python
# color image를 grey-scale로 변환해보기
# %reset

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

color_img = Image.open('./data/image/fruits.jpg')
plt.imshow(color_img)
plt.show()

color_pixel = np.array(color_img) # 3차원의 ndarray가 생성
print('color_pixel의 shape : {}'.format(color_pixel.shape)) # (426, 640, 3)
```

* 이미지 불러오는 방법은 동일
* 이미지는 기본 3차원이기 때문에 ndarray안에 이미지를 넣으면 3차원 ndarray가 생성
* shape을 찍어보면 3차원인 것을 알 수 있음



```python
# 흑백으로 처리하기
# 흑백 3차원으로 처리하기
# 각 pixel의 RGB값의 평균을 구해서 각각의 R,G,B값으로 설정
gray_pixel = color_pixel.copy()

for y in range(gray_pixel.shape[0]):
    for x in range(gray_pixel.shape[1]):
        gray_pixel[y,x] = int(np.mean(gray_pixel[y,x]))
        
plt.imshow(gray_pixel)
plt.show()

print(gray_pixel.shape) # (426, 640, 3)
```

* 3차원 이미지 흑백으로 전환하기 위해서 변수에 컬러 이미지 copy 넣기
* 이중 for문으로 shape에서 이미지 좌표계만 indexing.
* 좌표 안에 있는 각각의 RGB 데이터의 평균을 내어 다시 3개 동일하게 데이터 입히기.
* imshow로 보면 흑백으로 변환 확인.
* shape을 찍어보면 3차원은 유지하되 흑백으로 변환한 것을 확인.



```python
# grey-scale을 3차원이지만 2차원으로 변경가능 
# 왜냐하면 픽셀의 3개 값이 모두 같기 때문에 
# 하나만 남기고 모두 날려버리면 2차원

# 흑백 이미지 2차원 표현하기
gray_2d_pixel = gray_pixel[:,:,0] # 3개의 같은 값 중 1개만 가져오기
print(gray_2d_pixel.shape) # (426, 640)

# plt로 흑백해보기
plt.imshow(gray_2d_pixel, cmap='gray') # cmap 없이 하면 흑백이 안됨
plt.show()

# 흑백처리된 2차원 ndarray 이미지 파일 저장하기
gray_2d_image = Image.fromarray(gray_2d_pixel) # numpy array로부터 이미지 객체를 만들어주는 함수.
gray_2d_image.save('./data/image/my_gray_image.jpg')
```

* 흑백 3차원을 2차원으로 변환해보기.
* 3개의 같은 픽셀 값 중 2개를 제외시켜 변환하는 방법.
* [:, :, 0]는 행,열과  픽셀 안에 있는 데이터 중 [0] indexing으로 1개만 가져오기
* shape을 찍어보면 2차원으로 변환됨.
* plt imshow로 보려면 cmap을 지정해줘야함.





