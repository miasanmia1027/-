import numpy as np
import torch


m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print( m1.shape) # 2 x 2
print( m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1


#-------------------------------------------------------------------------------

#numpy일때
#t.shape==크기
#t.nidm==차원크기
#슬라이싱== 이것들은 어디서부터 어디까지를 알려준다

#torch일때
#t.dim==차원크기
#t.shape==t.size==크기
#슬라이싱==이것들은 특정 부분만 알려준다(print(t[:, 1]))  여기서 ':' 이거는 모든 것을 설정했다는 것을 알려준다
#   +a(t[:, :-1]) 뒤에 ':-1'이거는 마지막을 뺀 모든 것을 의미한다

#-------------------------------------------------------------------------------

#브로드캐스팅==원래 행렬 계산을 할때 좌우의 크기가 같아야 한다 그러나 프로그래밍에서 어쩔수 없이 크기가 다른 상황이 오는데 그것을 해결하기 위해 자동으로 변형해주는 기능
#이것이 자동으로 계산되는 방식== x,y둘다 필요한데 만약 하나 밖에 없으면 그냥 하나 있는 것을 복사하여 계산
#이것은 스칼라이든 백터이든 상관이 없다

#    브로드캐스팅 과정에서 실제로 두 텐서가 어떻게 변경되는지 보겠습니다.
# [1, 2]
# ==> [[1, 2],
#      [1, 2]]
# [3]
# [4]
# ==> [[3, 3],
#      [4, 4]]

#-------------------------------------------------------------------------------

#<행렬의 곱셈>
#행렬 곱셈(.matmul)과 원소 별 곱셈(.mul)



#m1=([[1, 2], [3, 4]])      m2=([[1], [2]])
#중요         m1.matmul(m2)==
# <첫 번째 요소>
# 1*1 + 2*2 = 1 + 4 = 5
# <두 번째 요소>
# 3*1 + 4*2 = 3 + 8 = 11



#m1=([[1, 2], [3, 4]])      m2=([[1], [2]])
#중요           m1.mul(m2)==m1 * m2
#<결과>
#[[1., 2.],     [6., 8.]]
#여기는 그냥 상식적으로 곱한것


#-------------------------------------------------------------------------------


#평균 구하기==t.mean
#mean에게는 dim이라는 차원을 더할수있다.
#  ex)t = torch.FloatTensor([[1, 2], [3, 4]])
#     print(t.mean(dim=0))   이러면 행을 ㅈ우고 열만 남긴다는 뜻 == 1과3의 평균,2와4의 평균값이 나오게된댜ㅏ.
#결과==tensor([2., 3.])
#+a='dim=1'이면 열을 지운다는 뜻

#-------------------------------------------------------------------------------


#덧셈== sum
#그냥 sum은 그냥 sum
#(t.sum(dim=0)) 이렇게 하면 행제거 dim=1이렇게 하면 열 제거

#-------------------------------------------------------------------------------

#최대(Max)와 아그맥스(ArgMax)
#max는 진짜 기본
#여기서도 dim이 들어가면 ex) dim=0 첫번째 차원 삭제
#(tensor([3., 4.]), tensor([1, 1])) 이따구로 출력이된다. 옆에 tensor([1, 1]) 이게 있는 이유는 Argmax 이것이다. 이것은 주어진 배열이나 함수에서 가장 큰 값을 가지는 인덱스를 반환한디.






