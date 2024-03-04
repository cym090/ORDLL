import os
import sys
import numpy as np
from copy import deepcopy
from sklearn.metrics import f1_score
import logging

def get_curve_online(known, novel, stypes = ['Bas']):
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

def metric_ood(x1, x2, stypes = ['Bas'], verbose=True):
    tp, fp, tnr_at_tpr95, th = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        # print('      ',)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
            # print(' {mtype:6s}'.format(mtype=mtype),)
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
            # print('{stype:5s} '.format(stype=stype),)
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            # print(' {val:6.3f}'.format(val=results[stype][mtype]))
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = 100.*(-np.trapz(1.-fpr, tpr))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            # print(' {val:6.3f}'.format(val=results[stype][mtype]))
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            # print(' {val:6.3f}'.format(val=results[stype][mtype]))
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            # print(' {val:6.3f}'.format(val=results[stype][mtype]))
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            # print(' {val:6.3f}'.format(val=results[stype][mtype]))
            print('')
    
    return results

def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values
    
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)] 

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n-1):
        CC = s_k_target[k+1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n+1] = 1.0
    FPR[n+1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):
        h =   ROC[j][0] - ROC[j+1][0]
        w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

        OSCR = OSCR + h*w

    return OSCR

def close_f1(klogits,ulogits,labels,known=10):
    k_pred = np.argmax(klogits,axis=1)
    u_pred = np.argmax(ulogits,axis=1)
    x1, x2 = np.max(klogits, axis=1), np.max(ulogits, axis=1)
    tp, fp, tnr_at_tpr95,th = get_curve_online(x1,x2)
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