import os
import sys
import numpy as np
from sklearn.metrics import f1_score
from copy import deepcopy
def get_curve_online2(known, novel, stypes = ['Bas']):
    '''
    konwn:闭集预测logit(max)
    novel:开集预测logit(max)
    '''
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]#闭集样本数
        num_n = novel.shape[0]#开集样本数
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]#th=novel[k]:>th,+1
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1 #th=known[k]:>th,+1
                    fp[stype][l+1] = fp[stype][l]
                    
        tpr90_pos = np.abs(tp[stype] / num_k - .90).argmin()
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()#tpr:正类样本被正确判为正类的比例，tpr95_pos:比例95%时阈值所在处
        tpr98_pos = np.abs(tp[stype] / num_k - .98).argmin()
        tpr99_pos = np.abs(tp[stype] / num_k - .99).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n#tnr:负类样本被正确分为负类的比例，95tpr处时的tnr
        whole_logits = np.sort((np.concatenate((known,novel))))
        th90 = whole_logits[tpr90_pos]
        th95 = whole_logits[tpr95_pos]
        th98 = whole_logits[tpr98_pos]
        th99 = whole_logits[tpr99_pos]
        th = {}
        th['90'] = th90
        th['95'] = th95
        th['98'] = th98
        th['99'] = th99
        
    return tp, fp, tnr_at_tpr95,th

def close_f1(klogits,ulogits,labels,known=10):
    k_pred = np.argmax(klogits,axis=1)
    u_pred = np.argmax(ulogits,axis=1)
    x1, x2 = np.max(klogits, axis=1), np.max(ulogits, axis=1)
    tp, fp, tnr_at_tpr95,th = get_curve_online2(x1,x2)
    P = ['90','95','98','99']
    score = {}
    for p in P:
        ktemp = x1 <= th[p]
        utemp = x2 <= th[p]
        kp = deepcopy(k_pred)
        up = deepcopy(u_pred)
        kp[ktemp] = known
        up[utemp] = known
        pred_labels = np.concatenate([kp,up],0)
        score[p] = f1_score(labels,pred_labels,average='macro')
        
    return score
        
        
        
    