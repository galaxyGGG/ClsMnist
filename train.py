import torch 
import torch.nn as nn
import numpy as np 
import sys

from torch import optim 

@torch.no_grad()
def caculateAcc(y_pred,y_true):
    _ , predIndex = torch.max(torch.softmax(y_pred,dim=1),dim=1)
    return 1 - torch.abs(predIndex - y_true).float().clamp(0,1).mean().item()
    
@torch.no_grad()
def test(net,testLoader,lossFun,device):
    losses = []
    accs = []
    for i ,(inputData , label) in enumerate(testLoader):
        inputData , label = inputData.to(device),label.to(device)
        output = net(inputData.to(device))
        loss = lossFun(output,label.to(device))
        losses.append(loss.item()) 
        accs.append(caculateAcc(output,label))
        print(
            "\rval:step:{:5d},testloss:{:5f},acc:{:5f}".format(
            i,losses[-1],accs[-1]
            ),end='')
        
    return {'loss':np.mean(np.array(losses)),'acc':np.mean(np.array(accs))}

def train(net,parameter,trainLoader,testLoader,lossFun=nn.CrossEntropyLoss(),
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    net.to(device)

    lossesTest = []
    optimizer = getattr(optim,parameter['optimizer'])(net.parameters(),lr=parameter['lr'])
    
    for epoch in range(parameter['epochs']):
        losses = []
        accs = []
        net.train()
        for i ,(image,label) in enumerate(trainLoader):
            image,label = image.to(device),label.to(device)
            optimizer.zero_grad()
            output = net(image)
            loss = lossFun(output,label)
            loss.backward() 
            optimizer.step()
            losses.append(loss.item()) 
            accs.append(caculateAcc(output,label))
            print(
                "\rtrain:step:{:5d},loss:{:5f},acc:{:5f}".format(
                i,losses[-1],accs[-1]
                ),end='')
        # print('\n')
        net.eval()
        testData = test(net,testLoader,lossFun=lossFun,device=device)
        lossesTest.append(testData['loss'])
        
        print(
            '\nepoch:{:3d},trainLoss:{:5f},testLoss:{:5f},trainAcc:{:5f},testAcc:{:5f}'.format(
            epoch,np.mean(np.array(losses)),lossesTest[-1],np.mean(np.array(accs)),testData['acc']
            ))
        
        if lossesTest[-1]<=np.array(lossesTest).min():
            torch.save(dict({'weight':net.state_dict()}, **parameter),
                       sys.path[0]+'/weight/'+parameter['modelName']+'_'+parameter['optimizer']+"_"+parameter["pkl_name"]+
                       '_lr'+str(parameter['lr'])+'_Seed'+str(parameter['seed'])+ "_epoch:{:3d}_trainAcc{:.2f}_testAcc{:.2f}".format(epoch,np.mean(np.array(accs)),testData['acc'])
                       +'.pkl')
            print('save')