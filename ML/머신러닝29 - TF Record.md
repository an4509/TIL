병목현상

일반적으로 프로그램이 병목현상을 겪게 되는데 그 중 가장 큰 요인 중 하나가 IO(Input/output)



tensorflow에서의 해결방법

tf record

다수의 데이터 파일을 하나의 파일로 압축하는 개념

tensorflow로 deep learning할 때 필요한 데이터를 보관하기 위한 자료구조 데이터 포맷

tensorflow의 표준 데이터 포맷 (binary 데이터 포맷)

입력데이터(x)와 label(t)를 하나의 파일에서 관리



과정

1. DataFrame 생성
2. TF Record 생성 함수 정의
3. data split
4. tf record 생성
5. tf record 읽어서 다시 복원