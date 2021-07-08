
from torch.utils.data import DataLoader
import train
import model
import loadDataMnist

parameter = {'lr':0.01,
             'batch_size':4,
             'epochs':400,
             'num_workers':4,
             'optimizer':'SGD',
             'seed':10000,
             'modelName':'ResNetReDefine',
             "pkl_name":"all_classes",
             }

def seedSet(seed):
    import os 
    import torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled=True
    
def main()->None:
    #net = model.FCNet()
    #net = model.ResNetReDefine()
    dataset = loadDataMnist.dataSplit()
    trainset = dataset['trainset']
    testset = dataset['testset']
    classes = trainset.classes
    numClasses = len(classes)
    parameter.update({'numClasses':numClasses,'classes':classes})
    net = getattr(model,parameter['modelName'])(outputShape = parameter['numClasses'])
    trainLoader = DataLoader(trainset,batch_size=parameter['batch_size'],num_workers=parameter['num_workers'],shuffle=True)
    testLoader = DataLoader(testset,batch_size=parameter['batch_size'],num_workers=parameter['num_workers'],shuffle=False)
    train.train(net,parameter,trainLoader,testLoader)
    
if __name__ == '__main__':
    seedSet(parameter['seed'])
    main()