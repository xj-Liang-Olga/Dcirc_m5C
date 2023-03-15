import numpy as np
import pandas as pd
import math
import argparse
from itertools import cycle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp.autocast_mode as autocast
from Bio import SeqIO
from Bio.Seq import Seq
import time
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score, \
    precision_recall_curve
from seq_load_one_hot_NCP_EIIP import *
# from resnetwithCBAM import *
from model_one_hot_NCP_EIIP import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, StratifiedKFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is {}'.format(device))

def save_checkpoint(state, is_best, model_path):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, model_path + '/' + 'checkpoint_mus_5_cv_test.pth.tar')

    else:
        print("=> Validation Performance did not improve")

def ytest_ypred_to_file(y_test, y_pred, out_fn):
    with open(out_fn,'w') as f:
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred[i])+'\n')

pos_train_fa = '/home/li/public/lxj/erdai_5mC/test_data/mus/after_cdhit_pos.fa'
neg_train_fa = '/home/li/public/lxj/erdai_5mC/test_data/mus/after_random_neg.fa'
X, y = load_train_val_bicoding(pos_train_fa, neg_train_fa)
# print(len(X), len(y))

model_path = '.'
vec_len = 8
HIDDEN_NUM = 128

LAYER_NUM = 3

RNN_DROPOUT = 0.5
FC_DROPOUT = 0.5
CELL = 'LSTM'
LEARNING_RATE = 0.001
BATCH_SIZE = 32

tprs = []
ROC_aucs = []
fprArray = []
tprArray = []
thresholdsArray = []
mean_fpr = np.linspace(0, 1, 100)

precisions = []
PR_aucs = []
recall_array = []
precision_array = []
mean_recall = np.linspace(0, 1, 100)

np.random.seed(12)
np.random.shuffle(X)
np.random.seed(12)
np.random.shuffle(y)


best_acc_list = []

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 / 8, random_state=42)
# print(X_train.shape, X_test.shape)
# print(X_train[0], X_train[1])
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=10).split(X, y)
for train_index, test_index in folds:
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1] / vec_len), vec_len)
    # print(X_train.shape)
    X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1] / vec_len), vec_len)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    X_test = torch.from_numpy(X_test).float()
    X_train = X_train.reshape(X_train.shape[0], vec_len, X_train.shape[1])
    # print(X_train.shape)
    X_test = X_test.reshape(X_test.shape[0], vec_len, X_test.shape[1])
    # print(X_train.shape, X_test.shape, type(X_train))

    # X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    model = CNN65_RNN(HIDDEN_NUM, LAYER_NUM, FC_DROPOUT, RNN_DROPOUT, CELL)
    # model = ronghe(HIDDEN_NUM, LAYER_NUM, FC_DROPOUT, CELL)
    model.to(device)
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    # loss = loss.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.100000000000000002)

    best_acc = 0
    patience = 0
    def train(model, loss, optimizer, x_val, y_val):
        x = x_val.to(device)
        y = y_val.to(device)
        model.train()
        optimizer.zero_grad()

        # with torch.cuda.amp.autocast():


        fx = model(x)
        loss = loss.forward(fx,y)
        pred_prob = F.log_softmax(fx, dim=1)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        # scaler.scale(output).backward()
        # scaler.step(optimizer)
        # scaler.update()
        #

        return loss.cpu().item(), pred_prob, list(np.array(y_val.cpu())), list(fx.data.cpu().detach().numpy().argmax(axis=1))#cost,pred_probability and true y value

    def predict(model, x_val):
        model.eval()
        x = x_val.to(device)

        with torch.no_grad():
            fx = model(x)

        return fx

    EPOCH = 15
    n_classes = 2
    n_examples = len(X_train)

    Loss_list = []
    Accuracy_list = []

    # scaler = torch.cuda.amp.GradScaler()

    for i in range(EPOCH):

        start_time = time.time()

        cost = 0.
        y_pred_prob_train = []
        y_batch_train = []
        y_batch_pred_train = []

        num_batches = n_examples // BATCH_SIZE
        for k in range(num_batches):
            start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
            # X_train[start:end] , y_train[start:end] = X_train[start:end].to(device), y_train[start:end].to(device)
            # print(X_train[start:end].shape, y_train[start:end].shape)
            # print(X_train.shape)
            output_train, y_pred_prob, y_batch, y_pred_train = train(model, loss, optimizer, X_train[start:end], y_train[start:end])
            cost += output_train

            prob_data = y_pred_prob.cpu().detach().numpy()
            # prob_data = y_pred_prob.data.numpy()
            # if args.if_bce == 'Y':
            #     for m in range(len(prob_data)):
            #         y_pred_prob_train.append(prob_data[m][0])

            # else:
            for m in range(len(prob_data)):
                y_pred_prob_train.append(np.exp(prob_data)[m][1])

            y_batch_train += y_batch
            y_batch_pred_train += y_pred_train

        scheduler.step()

        #rest samples
        start, end = num_batches * BATCH_SIZE, n_examples
        output_train, y_pred_prob, y_batch, y_pred_train = train(model, loss, optimizer, X_train[start:end], y_train[start:end])
        cost += output_train


        prob_data = y_pred_prob.cpu().detach().numpy()
        # prob_data = y_pred_prob.data.numpy()
        # if args.if_bce == 'Y':
        #     for m in range(len(prob_data)):
        #         y_pred_prob_train.append(prob_data[m][0])


        for m in range(len(prob_data)):
            y_pred_prob_train.append(np.exp(prob_data)[m][1])


        y_batch_train += y_batch
        y_batch_pred_train += y_pred_train

        # train AUC
        fpr_train, tpr_train, thresholds_train = roc_curve(y_batch_train, y_pred_prob_train)

        # predict test
        output_test = predict(model, X_test)
        y_pred_prob_test = []

        # if args.if_bce == 'Y':
        #     y_pred_test = []
        #     prob_data = F.sigmoid(output_test).data.numpy()
        #     for m in range(len(prob_data)):
        #         y_pred_prob_test.append(prob_data[m][0])
        #         if prob_data[m][0] >= 0.5:
        #             y_pred_test.append(1)
        #         else:
        #             y_pred_test.append(0)
        # else:
        y_pred_test = output_test.data.cpu().numpy().argmax(axis=1)
        # y_pred_test = output_test.data.numpy().argmax(axis=1)
        # print(y_pred_test)
        prob_data = F.log_softmax(output_test, dim=1).data.cpu().numpy()
        # prob_data = F.log_softmax(output_test, dim=1).data.numpy()
        # print(prob_data)
        for m in range(len(prob_data)):
            y_pred_prob_test.append(np.exp(prob_data)[m][1])

        # test AUROC
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_prob_test)

        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)



        print(
            "Epoch %d, cost = %f, AUROC_train = %0.4f, train_acc = %.4f, train_recall= %.4f,train_precision = %.4f, train_f1score = %.4f,train_mcc= %.4f, test_acc = %.4f, test_recall= %.4f,test_precision = %.4f, test_f1score = %.4f,test_mcc= %.4f,AUROC_test = %0.4f"
            % (i + 1, cost / num_batches, auc(fpr_train, tpr_train), accuracy_score(y_batch_train, y_batch_pred_train),
               recall_score(y_batch_train, y_batch_pred_train), precision_score(y_batch_train, y_batch_pred_train),
               f1_score(y_batch_train, y_batch_pred_train),
               matthews_corrcoef(y_batch_train, y_batch_pred_train), accuracy_score(y_test, y_pred_test),
               recall_score(y_test, y_pred_test), precision_score(y_test, y_pred_test), f1_score(y_test, y_pred_test),
               matthews_corrcoef(y_test, y_pred_test),
               auc(fpr_test, tpr_test)))

        print("time cost: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        cur_acc = accuracy_score(y_batch_train, y_batch_pred_train)
        is_best = bool(cur_acc >= best_acc)
        best_acc = max(cur_acc, best_acc)

        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_acc,
            'optimizer': optimizer.state_dict()
        }, is_best, model_path)

        if not is_best:
            patience += 1
            if patience >= 5:
                break

        else:
            patience = 0

        if is_best:
            ytest_ypred_to_file(y_batch_train, y_pred_prob_train,
                                model_path + '/' + 'predout_train.tsv')

            ytest_ypred_to_file(y_test, y_pred_prob_test,
                                model_path + '/' + 'predout_val.tsv')

    fprArray.append(fpr_test)
    tprArray.append(tpr_test)
    thresholdsArray.append(thresholds_test)
    tprs.append(np.interp(mean_fpr, fpr_test, tpr_test))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr_test, tpr_test)
    ROC_aucs.append(roc_auc)

    recall_array.append(recall_test)
    precision_array.append(precision_test)
    precisions.append(np.interp(mean_recall, recall_test[::-1], precision_test[::-1])[::-1])
    pr_auc = auc(recall_test, precision_test)
    PR_aucs.append(pr_auc)

colors = cycle(['#5f0f40', '#9a031e', '#fb8b24', '#e36414', '#0f4c5c'])
## ROC plot for CV
fig = plt.figure(0)
for i, color in zip(range(len(fprArray)), colors):
    plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.9, color=color,
             label='ROC fold %d (AUC = %0.4f)' % (i + 1, ROC_aucs[i]))
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#c4c7ff',
#          label='Random', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
ROC_mean_auc = auc(mean_fpr, mean_tpr)
ROC_std_auc = np.std(ROC_aucs)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('/home/li/public/lxj/erdai_5mC/test_data/mus/train_roc.png')
plt.close(0)

fig = plt.figure(1)
for i, color in zip(range(len(recall_array)), colors):
    plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.9, color=color,
             label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, PR_aucs[i]))
mean_precision = np.mean(precisions, axis=0)
mean_recall = mean_recall[::-1]
PR_mean_auc = auc(mean_recall, mean_precision)
PR_std_auc = np.std(PR_aucs)

# plt.plot(mean_recall, mean_precision, color='#ea7317',
#          label=r'Mean PRC (AUPRC = %0.2f $\pm$ %0.2f)' % (PR_mean_auc, PR_std_auc),
#          lw=1.5, alpha=.9)
# std_precision = np.std(precisions, axis=0)
# precision_upper = np.minimum(mean_precision + std_precision, 1)
# precision_lower = np.maximum(mean_precision - std_precision, 0)
# plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
#                  label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.savefig('/home/li/public/lxj/erdai_5mC/test_data/mus/train_pr.png')
plt.close(0)

best_acc_list.append(best_acc)
print('> best acc:', best_acc)