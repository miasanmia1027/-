import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# # 훈련 데이터
# x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
# x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
# x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# # 가중치 w와 편향 b 초기화
# w1 = torch.zeros(1, requires_grad=True)
# w2 = torch.zeros(1, requires_grad=True)
# w3 = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)


# # optimizer 설정
# optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

# nb_epochs = 1000
# for epoch in range(nb_epochs + 1):

#     # H(x) 계산
#     hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

#     # cost 계산
#     cost = torch.mean((hypothesis - y_train) ** 2)
# #이 부분도 이해가 안된다 어떻게 해야하까?
# #이 부분은 손실 함수를 의미한다.





#     # cost로 H(x) 개선
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()

# #이 부분이 아직도  이해가 안된다 이거를 어떻게 해야 할까?



#     # 100번마다 로그 출력
#     if epoch % 100 == 0:
#         print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
#             epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
#         ))







# 백터의 내적을 이용하여 hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b   이렇게 긴것을 간단하게   hypothesis = x 이렇게 바꿀수있다.










x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
# 여기서 특이한 것은 가중치가 배터인것이다. 이때까지는 스칼라만 있었는데




 
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))




















