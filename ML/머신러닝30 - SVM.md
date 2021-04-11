# SVM

**개념**

* 성능은 Deep Learning에 비해 살짝 떨어짐

* Decision Boundaries (결정경계)라는 개념을 이용해서 분류를 위한 기준선을 만들어서 학습

* 결정경계가 중간에 위치한게 좋은 모델



## Support vector

**개념**



찾기

vector 연산을 통해 찾음



알고리즘

suport vector을 통과하는 가상의 직선을 앞 쪽의 선과 평행이 되도록 그음

이 두개의 가운데 점을 margin값을 최대로 하는 경계가 결정경계



특성

장점 : 속도가 상당히 빠름.



보통 sklearn에서 잘 구현한 svm 사용



주의할 점

이상치 처리에 주의해야함.

margin값이 작아지게 되고 과대적합이 일어남 -> 하드margin

반대로 margin값이 큰 것은 soft margin



이상치가 섞여있는 경우 선형으로 데이터를 분리하기 힘들어짐.

이 문제를 해결하기 위해 데이터 오류를 허용하는 전략이 만들어짐 -> Regularization



sklearn에서 svm c(cost) hyper parameter : 얼마나 많은 데이터 포인트가 다른 범위에 놓이는 것을 허용하는 정도

기본값 1이고 이 값이 클수록 다른 범주에 놓이는 데이터 포인트를 적게 허용 

c값을 크게 잡으면 마진을 적게 잡음 -> 과대적합

c값을 적게 잡으면 마진을 크게 잡음 -> 과소적합



데이터가 섞여있는 경우

kernel이라는 hyper parameter에 linear poly로 해서 차원을 3차원으로 사상시키면 데이터가 분리됨.

구분 짓는 평면을 만들 수 있음.

결정경계는 무조건 선이 아니라 면이될 수도 있음.



poly -> polynomial

rbf -> Radial bias function 방사기제 함수. 고차원으로 사상시키는 함수

rbf는 모든 상황을 포함하기 때문에 보통 rbf사용

가우시만 커널이라고 함. (기본값)



차원을 사상시키는 함수를 사용하려면 gamma라는 hyper parameter를 지정

결정경계를 얼마나 유연하게 그릴것인지를 의미

gamma가 작으면 직선에 가깝게 그려지고 -> 과소적합

gamma가 크면 구불구불하게 그려짐 -> 과대적합



svm model를 만드는 것은 어렵지 않으나 Hyper parameter를 조절해서 최적화하는 것이 어려움

수동으로 변경 실행하기에 어려워서 자동화 시켜주는 것들

sklearn

Grid search cv(cross validation) : hyper parameter의 값을 몇개 정해줌

Randaomise cv : 범위를 지정하고 랜덤하게 추출해서 cv를 실행