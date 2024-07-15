import torch
import torch.nn as nn
import torch.nn.functional as F


# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])
# # 정보 주기


# # 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
# model = nn.Linear(1,1)
# #모델은 선형회귀이다. 그리고 차원은 1,1이다 즉 들어가는 값과 나온느 값의 차우너이 1이라는 의미이다.
# #즉 input은 x_train이고 output은 y_train이다. 

# print(list(model.parameters()))
# #이것은 가중치w와b를 선언 하는 것이다.
# #<결과>
# # [Parameter containing:
# # tensor([[0.1443]], requires_grad=True), Parameter containing:   이것은 w
# # tensor([-0.5163], requires_grad=True)]  이것은 b 이다.
# #requires_grad=True== 학습 대상이라는 의미


# # optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
# #parameter의 의미== 이것은 학습하면서 최적의 함수를 만들기 위해 계속 업데이트를 시켜주는 관리자 느낌

# #------------------------------------------------------------------------------------------------------------------

# # 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
# nb_epochs = 2000
# for epoch in range(nb_epochs+1):

#     # H(x) 계산
#     prediction = model(x_train)
#     #모델은 선형 회귀를 쓴다는의미 앞에서 model을 정의 했기 때문에


#     # cost 계산
#     cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

#     # cost로 H(x) 개선하는 부분
#     # gradient를 0으로 초기화
#     optimizer.zero_grad()
#     # 비용 함수를 미분하여 gradient 계산
#     cost.backward() # backward 연산
#     # W와 b를 업데이트
#     optimizer.step()

#     if epoch % 100 == 0:
#     # 100번마다 로그 출력
#       print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#           epoch, nb_epochs, cost.item()
#       ))
# print(list(model.parameters()))















#---------------------------------------------------------------------------------------------------


# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3,1)
#들어가는 x_train이 3차원이여서 3이다.

print(list(model.parameters()))


optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

nb_epochs = 200000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))













