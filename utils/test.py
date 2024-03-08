import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import f1_score
from utils import evaluation
from utils.thf1 import close_f1
def test(model, criterion, testloader, outloader, options,return_logits=False):
    '''
    correct:闭集预测真确数
    total:总样本量
    '''
    num_classes = len(options.dataset.known_class)
    model.eval()
    close_correct,open_correct, total = 0,0, 0
    results = {}
    torch.cuda.empty_cache()

    close_k_logits, close_u_logits, close_labels,close_labels,all_labels = [], [], [],[],[]#_pred_k:闭集数据集logit,_pred_u:开集数据logit,close_labels:真实标签,o_
    open_k_logits,open_u_logits = [],[]
    close_k_pred , open_k_pred= [],[]
    open_u_pred = []
    
    with torch.no_grad():
        for data, labels in testloader:#闭集分类
            data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                feat = model(data)
                logits = criterion(feat, model.centers, model.radius1, model.radius2, ) #(batch_size,num_classes)
                total += labels.size(0)
                close_labels.append(labels.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
                
                if options['open_test'] == True:
                    close_pred = logits[:,:-1].data.max(1)[1]
                    open_pred = logits.data.max(1)[1]
                    
                    close_k_pred.append(close_pred.data.cpu().numpy())
                    open_k_pred.append(open_pred.data.cpu().numpy())
                    
                    close_k_logits.append(logits[:,:-1].data.cpu().numpy())
                    open_k_logits.append(logits.data.cpu().numpy())
                        
                    close_correct += (close_pred == labels.data).sum()
                    open_correct += (open_pred == labels.data).sum()
                    
                else:
                    pred = logits.data.max(1)[1]
                    close_k_pred.append(pred.data.cpu().numpy())
                    close_k_logits.append(logits.data.cpu().numpy())
                    close_correct += (pred == labels.data).sum()
                    
        for batch_idx, (data, labels) in enumerate(outloader):#unkonwn dection
            data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                feat = model(data)
                logits = criterion(feat, model.centers, model.radius1, model.radius2, )
                dummy_labels = np.full((feat.size(0)),num_classes)
                all_labels.append(dummy_labels)
                if options['open_test'] == True:
                    pred = logits.data.max(1)[1]
                    open_u_pred.append(pred.data.cpu().numpy())
                    close_u_logits.append(logits[:,:-1].data.cpu().numpy())
                    open_u_logits.append(logits.data.cpu().numpy())
                else:
                    # pred = logits.data.max(1)[1]
                    close_u_logits.append(logits.data.cpu().numpy())
    # Accuracy
    if options['open_test'] == True:
        close_acc = float(close_correct) * 100. / float(total)
        open_acc = float(open_correct) * 100. / float(total)
        results['close_acc'] = close_acc
        results['open_acc'] = open_acc
        print('Close Acc: {:.3f},Open Acc:{:.3f}'.format(close_acc,open_acc))
    else:
        close_acc = float(close_correct) * 100. / float(total)
        results['close_acc'] = close_acc
        print('Close Acc: {:.3f}'.format(close_acc))
        
    close_k_logits = np.concatenate(close_k_logits, 0) #(close_num_samples,num_classes)
    close_u_logits = np.concatenate(close_u_logits, 0) #(open_num_samples,num_classes)
    close_labels = np.concatenate(close_labels, 0) #(close_num_samples,)
    all_labels = np.concatenate(all_labels,0)
    
    if options['open_test'] == True:
        open_k_logits = np.concatenate(open_k_logits, 0)#(close_num_samples,num_classes+1)
        open_k_pred = np.concatenate(open_k_pred,0)
        open_u_logits = np.concatenate(open_u_logits, 0)#(open_num_samples,num_classes+1)
        open_u_pred = np.concatenate(open_u_pred,0)#(open_num_samples,)
        
        open_pred = np.concatenate([open_k_pred,open_u_pred],0)
        
        f1 = f1_score(all_labels,open_pred,average='macro')
        results['f1'] = f1
   
    th_f1 = close_f1(close_k_logits,close_u_logits,all_labels,num_classes)
    results.update(th_f1)
    # Out-of-Distribution detction evaluation
    # x1:已知类最大logits,x2:未知类最大Logits
    x1, x2 = np.max(close_k_logits, axis=1), np.max(close_u_logits, axis=1)
    print('Close OOD Metric:')
    out_metric = evaluation.metric_ood(x1, x2)['Bas'] #在95%正类样本被识别为正类时时:TP,TN,
    results.update(out_metric)
    _oscr_socre = evaluation.compute_oscr(close_k_logits, close_u_logits, close_labels)
    results['close_oscr'] = _oscr_socre * 100.
    
    if options['open_test'] == True:
        x1, x2 = np.max(open_k_logits, axis=1), np.max(open_u_logits, axis=1)
        print('Open OOD Metric:')
        open_out_metric = evaluation.metric_ood(x1, x2)['Bas']
        results['open_out_metric'] = open_out_metric
        # open_oscr_socre = evaluation.compute_oscr(open_k_logits, open_u_logits, all_labels)
        # results['open_oscr'] = open_oscr_socre*100
    if return_logits:
        return results, open_k_logits, open_u_logits
    return results