# CNN 개요

> Convolutional Neural Network
>
> 합성곱 신경망





## 들어기전 용어정리

**DNN(Deep Neural Network)**

* 일반적인 딥러닝의 구조

![](md-images/DNN_%EA%B5%AC%EC%A1%B0.PNG)

* Input, hidden, output 각각의 Layer로 구성
* 내부에는 w,b와 relu함수 적용해서 최종적으로 softmax함수사용해서 나온 예측값을 정답과 비교(cross entropy)
* 반복학습간 feed forward와 back propagation이 적용.



**FC Layer**

* Fully Connected Layer
* 일반적으로 DNN과 같은 의미로 혼용해서 사용하지만 잘 못 된 것.
* 이전 Layer와 이후 Layer들의 node들이 완전히 연결된 상태.
* Dens Layer가 FC Layer.





## CNN 개념

* Pixel에 대한 정보를 학습하면서 이미지의 특징을 학습
* 이미지를 분류하기 위해 이미지의 pattern을 이용는 딥러닝 방법론(알고리즘)
* 알고리즘을 이용해서 각 이미지가 가지고 있는 패턴을 추출해서 학습시키는 것이 중요.
* 즉 이미지 자체를 학습이 아니라 추출된 이미지 특징을 학습
* FC Layer로만 구성된 DNN의 입력데이터는 행렬곱 연산을 위해 1차원으로 한정 
  * 1차원으로 풀어주기 위해서 Flatten() layer을 사용
  * CNN이 나오기 전에는 이미지 인식을 우리가 했던 MNIST와 같은 2차원 데이터를 1차원으로 변형시켰던 방식.
  * 이미지 형상은 고려하지 않고 많은 양의 데이터를 직접 이용해서 학습 -> 시간이 오래걸림.
* 데이터 전처리 과정이 달리지지 전체적이 알고리즘이 크게 달라지지는 않음.





![](md-images/CNN_%EA%B5%AC%EC%A1%B0.PNG)

* w, b는 우리가 기존에 사용했던 개념이 아님.
* pooling 작업 진행해서 나온 결과가 다시 conv 작업 (반드시 나오는 것이 아님 어디든 나 올 수 있음)
* 다시 pooling을 할 수도 있고 conv로 들어갈 수도 있음. 
* hidden layer를 써도 되고 안써도 되는데 일반적으로 쓰지는 않음.
* 이후 DNN구조가 붙여짐.





***다음 강의 때는 각각의 구조에 대한 개념과 원리를 하나씩 알아보고 이해하기***







