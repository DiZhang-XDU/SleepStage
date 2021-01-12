import torch,os
from torch.autograd import Variable
import random
from tqdm import tqdm
import numpy as np
from tools.data_loader import XY_dataset_5inOne
from tools.tools import SelectBest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score
from sklearn import preprocessing as skp
import pickle

os.environ['CUDA_VISIBLE_DEVICES']= '0'

def tester(weightDir = 'weights/checkpoint_', myDataset = XY_dataset_5inOne, alpha = 0.5, scaler = None):
    saveObj = torch.load(weightDir)
    model = saveObj['net'].cuda()

    testSet = myDataset('test', datasetName='SHHS')    
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = 750, 
                                    shuffle = False, num_workers = 8, drop_last = False)

    model.eval()
    name = ['test']
    epoch_loss = {i:0 for i in name}; epoch_acc = {i:0 for i in name}
    record_target = {i:torch.LongTensor([]) for i in name}; record_pred = {i:torch.LongTensor([]) for i in name}

    # scaler
    if scaler is not None:
        ssd = skp.StandardScaler() 

    with torch.no_grad():
        tq = tqdm(testLoader, desc= 'Test', ncols=70, ascii=True)
        for i, (data, target, loc) in enumerate(tq):
            # scaler
            if scaler is not None:
                scaledData = torch.empty_like(data)
                for _ in range(data.shape[0]):
                    dummy = data[_].view([3750*5, 5])
                    ssd = ssd.partial_fit(dummy)      
                    dummy = ssd.transform(dummy)
                    dummy = torch.tensor(scaler.inverse_transform(dummy)).cuda()
                    dummy = dummy.reshape([5, 3750, 5])
                    scaledData[_,:,:,:] = dummy
                data = scaledData

            inputs = Variable(data.cuda())
            targets = Variable(target.cuda())
            x1, x2, loss1, loss2 = model(inputs, targets.view([-1]).long())
            pred = alpha * x1 + (1 - alpha) * x2
            loss = alpha * loss1 + (1 - alpha) * loss2

            #record
            pred = torch.argmax(pred, 1).cpu()
            record_pred['test'] = torch.cat([record_pred['test'], pred])
            record_target['test'] = torch.cat([record_target['test'], target])

            epoch_loss['test'] += loss.item()
            epoch_acc['test'] += accuracy_score(target, pred) 
            tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['test'] / (i + 1)), 
                        'Acc:':'{:.4f}'.format(epoch_acc['test'] / (i + 1))})
        epoch_loss['test'] /= (i + 1)

    epoch_acc['test'] = accuracy_score(record_target['test'], record_pred['test']) 
    msg_loss = 'Tst Loss:{:.4f}, acc:{:.2f}\n'.format(
                epoch_loss['test'],  epoch_acc['test']  * 100)
    msg_test_detail = classification_report(record_target['test'], record_pred['test'], labels=[0,1,2,3,4]) \
                                + str(confusion_matrix(record_target['test'], record_pred['test'], labels=[0,1,2,3,4])) \
                                + str(cohen_kappa_score(record_target['test'], record_pred['test'], labels=[0,1,2,3,4])) + '\n\n'
    print(msg_loss[:-1] + msg_test_detail)
    return msg_loss, msg_test_detail

# !!!!!!!!!!!!!!!!!
import multiprocessing
multiprocessing.set_start_method('spawn', True)


if __name__ == "__main__":
    prep = False
    ss = None

    if prep:
        prepdir = './prepared_data/prep.pkl'
        if not os.path.exists(prepdir):
            SHHSTrainSet = XY_dataset_5inOne('train',datasetName='SHHS')
            SHHSTrainLoader = torch.utils.data.DataLoader(SHHSTrainSet, batch_size = 1, 
                                    shuffle = False, num_workers = 8, drop_last = False)
            ss = skp.StandardScaler() 
            tq = tqdm(SHHSTrainLoader, desc= 'SHHS Scaling', ncols=70, ascii=True)
            for i, (data, target, loc) in enumerate(tq):
                dummy = data[:,2,:,:].view([3750, 5])
                ss.partial_fit(dummy)

            with open(prepdir, 'wb') as f:
                pickle.dump({'ss':ss}, f)
        else:
            with open(prepdir, 'rb') as f:
                cache = pickle.load(f)
            ss = cache['ss']
    
    alpha = .5
    msg0 = 'ep = {}, alpha = {}'.format(ep, alpha)
    print(msg0)
    msg1,msg2 = tester(alpha = alpha, scaler = ss)
    with open('history/log_tester.txt', 'a') as f:
        f.write(msg0 + '\n' + msg1 + msg2)