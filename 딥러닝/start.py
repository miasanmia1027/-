#미니 밴치 압도적인 데이커를 감당하기 위해 나온 방법
# 그냥 병렬 작업을 하는 느낌

#에포크(Epoch)== 전체 훈련 데이터가 학습에 한 번 사용된 주기

#미니밴치크기==배치크기

#<정리>
#미니 배치 경사 하강법== 미니 배치 단위로 경사 하강법을 하는 것
#[특징]가중치 최적값 잘찾음 but 계산량이 너무 큼

#배치 경사 하강법== 전체데이터를 한번에 하는 방법
#[특징]가중치 최적화하는데 해맴 but 계산량이 적음



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더


x_train= torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])
y_train= torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

data_set= TensorDataset(x_train,y_train)

dataloader= DataLoader(data_set,batch_size=2,shuffle=True)
#shuffle=True 이것은 epoch가 다음으로 넘어갈떄마다 batch 순서를 바꾸는 것
#batchsize는 2의 제곱수로 표현한다 //RE== 메모리가 2의 배수여서

model=nn.Linear(3,1)
optimizer= torch.optim.SGD(model.parameters(),lr=1e-5)

reapeat_number=20

for i in range(reapeat_number+1):
    for batch_idx,samples in enumerate(dataloader):
        x_train,y_train=samples

        predctin= model(x_train)

        cost=F.mse_loss(predctin,y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()        

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        i, reapeat_number, batch_idx+1, len(dataloader),
        cost.item()
        ))




#------------------------------------------------------------------------------------------------------------------








































