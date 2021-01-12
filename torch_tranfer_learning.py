import os,time
import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score

from tqdm import tqdm
from tensorboardX import SummaryWriter

from tools.data_loader import XY_dataset_5inOne as myDataset
from sleep_models.Net_ResV1_TwoLoss import Net_Seq_E2E as myNet


def trainer(resume = False, freq = 125):
    path = ('./weights', './history')
    for p in path:
        if os.path.exists(p) is False:
            os.mkdir(p) 
    EPOCH_NUM_MAX = 4
    EPOCH_STEP_NUM = 100
    BATCH_NUM = 100
    
    # !!!!!!!!!!!!!!!!! DataLoader 
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    # tensorboard
    import shutil,random
    if not resume:
        shutil.rmtree('logs')
    writer = SummaryWriter('logs')
    
    # dataloader prepare
    trainSet = myDataset('train', frame_len = 30*freq)   # 135573 -> 4659965(0.8) -> 5241551(0.9)
    validSet = myDataset('valid', frame_len = 30*freq)      # 581663 -> 291843(0.9)
    testSet = myDataset('test', frame_len = 30*freq)        # 584004
    # TVT scale: mass13-10328/10328/99592(0.1/0/0.9)                                        47977/950/61943 (50-left)   19737
    # TVT scale: CCSHS -136858/10328/551454(0.2/0/0.8)  344696/2882/343616(0.5/0.01/0.5)    66270/855/622042(50-left)   27335(50-left)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = BATCH_NUM, 
                                    shuffle = True, num_workers = 6, drop_last = False)
    validLoader = torch.utils.data.DataLoader(validSet, batch_size = BATCH_NUM * 2, 
                                    shuffle = False, num_workers = 6, drop_last = False)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size = BATCH_NUM * 2, 
                                    shuffle = False, num_workers = 6, drop_last = False)

    # options
    if resume:
        loadObj = torch.load('./weights/checkpoint')
        model, epoch, optim, scheduler, best_loss_val = loadObj['net'], loadObj['epoch'], loadObj['optim'], loadObj['sched'], loadObj['best_loss_val']
        epoch += 1
    else:
        # model = myNet(5).cuda()
        best_loss_val, epoch = 9999, 1
        model = torch.load('./weights/checkpoint_')['net']###############################
        
        ########modify########
        model.drop = nn.Dropout(.75).cuda()
        model.net.requires_grad = False
        model.bn1.requires_grad = False
        model.gap.requires_grad = False
        model.criterion2 = nn.CrossEntropyLoss(weight = torch.FloatTensor([1/25, 1/5, 1/40, 1/15, 1/15, 0])).cuda()
 
        optim = torch.optim.Adam(model.parameters(), lr= 1e-3)      # CC建议参数：1e-4,it=4,bs=50,lrd=2,.1
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,4,2e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 2, .1)


    print('start epoch')
    step = 0
    trainIter = iter(trainLoader)   
    for epoch in range(epoch, EPOCH_NUM_MAX + 1): 
        tic = time.time()
        alpha = .5
        # alpha = 1 - (epoch / EPOCH_NUM_MAX) ** 2
        name = ('train', 'valid', 'test')
        epoch_loss = {i:0 for i in name}
        epoch_acc = {i:0 for i in name}
        record_target = {i:torch.LongTensor([]) for i in name}
        record_pred = {i:torch.LongTensor([]) for i in name}

        torch.cuda.empty_cache()
        model.train()
        tq = tqdm(range(EPOCH_STEP_NUM), desc= 'Trn', ncols=75, ascii=True)
        for i, _ in enumerate(tq):
            data, target, loc = next(trainIter)
            step += 1
            if step == len(trainLoader):
                step = 0
                trainIter = iter(trainLoader)

            inputs = Variable(data.cuda())
            inputs.requires_grad = True
            targets = Variable(target.cuda())

            # forward
            x1, x2, loss1, loss2 = model(inputs, targets.view([-1]).long())
            pred = alpha * x1 + (1 - alpha) * x2
            loss = alpha * loss1 + (1 - alpha) * loss2
            # loss = loss2
            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # record
            pred = torch.argmax(pred,1).cpu()
            record_pred['train'] = torch.cat([record_pred['train'], pred])
            record_target['train'] = torch.cat([record_target['train'], target])
            
            epoch_loss['train'] += loss.item()
            epoch_acc['train'] += accuracy_score(target, pred) 
            tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss['train'] / (tq.n+1)), 
                            'Acc:':'{:.4f}'.format(epoch_acc['train'] / (tq.n+1))})
        epoch_loss['train'] /= (i+1)

        # eval
        torch.cuda.empty_cache()
        model.eval()
        valtestLoader = {'valid':validLoader, 'test':testLoader}
        for idx in valtestLoader:
            tq = tqdm(valtestLoader[idx], desc = {'valid':'Val','test':'Tst'}[idx], ncols=75, ascii=True)
            for i, (data, target, loc) in enumerate(tq):
                inputs = Variable(data.cuda())
                # inputs.requires_grad = True
                targets = Variable(target.cuda())
                with torch.no_grad(): 
                    x1, x2, loss1, loss2 = model(inputs, targets.view([-1]).long())
                alpha = 0.5
                pred = alpha * x1 + (1 - alpha) * x2
                loss = alpha * loss1 + (1 - alpha) * loss2
                
                #record
                pred = torch.argmax(pred,1).cpu()
                record_pred[idx] = torch.cat([record_pred[idx], pred])
                record_target[idx] = torch.cat([record_target[idx], target])

                epoch_loss[idx] += loss.item()
                epoch_acc[idx] += accuracy_score(target, pred) 
                tq.set_postfix({'Loss':'{:.4f}'.format(epoch_loss[idx] / (i+1)), 
                            'Acc:':'{:.4f}'.format(epoch_acc[idx] / (i+1))})
            epoch_loss[idx] /= (i+1)
        
        # epoch end
        scheduler.step()
        for idx in name:
            epoch_acc[idx] = accuracy_score(record_target[idx], record_pred[idx]) 

        msg_epoch = 'epoch:{:02d}, time:{:2f}\n'.format(epoch, time.time() - tic)
        msg_loss = 'Trn Loss:{:.4f}, acc:{:.2f}  Val Loss:{:.4f}, acc:{:.2f}  Tst Loss:{:.4f}, acc:{:.2f}\n'.format(
            epoch_loss['train'], epoch_acc['train'] * 100, 
            epoch_loss['valid'], epoch_acc['valid'] * 100,
            epoch_loss['test'],  epoch_acc['test']  * 100)
        msg_test_detail = classification_report(record_target['test'], record_pred['test'], labels=[0,1,2,3,4]) \
                                 + str(confusion_matrix(record_target['test'], record_pred['test'], labels=[0,1,2,3,4])) + '\nkappa: ' \
                                 + str(cohen_kappa_score(record_target['test'], record_pred['test'], labels=[0,1,2,3,4])) + '\n\n'
        print(msg_epoch + msg_loss[:-1] + msg_test_detail)

        # save
        writer.add_scalars('loss',{'train':epoch_loss['train'] , 'valid':epoch_loss['valid'], 'test':epoch_loss['test']},epoch)
        writer.add_scalars('acc',{'train':epoch_acc['train'], 'valid':epoch_acc['valid'], 'test':epoch_acc['test']},epoch)
        with open('history/log.txt','a') as f:
            f.write(msg_epoch)
            f.write(msg_loss)
            f.write(msg_test_detail)
        # if best_loss_val > epoch_loss['valid']:
        if True:
            best_loss_val = epoch_loss['valid']
            saveObj = {'net': model, 'epoch':epoch, 'optim':optim , 'best_loss_val':best_loss_val}
            torch.save(saveObj, 'weights/epoch_{:02d}_val_loss={:4f}_acc={:.4f}'.format(epoch, epoch_loss['valid'], epoch_acc['valid']))
            torch.save(saveObj, 'weights/checkpoint')
    writer.close()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']= '0'
    trainer()