import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


def train(train_iter, test_iter, dev_iter, model, args, text_field, aspect_field, sm_field, predict_iter):
    time_stamps = []

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)

    steps = 0
    model.train()
    start_time = time.time()
    lr_now = args.lr
    b_acc = 0
    early = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, aspect, target = batch.text, batch.aspect, batch.sentiment
            temp_m = feature.permute(1,0).data.numpy()
            sen_mask = np.empty(temp_m.shape, dtype=np.float32)
            sen_mask[temp_m!=1] = 0.
            sen_mask[temp_m==1] = -9.9e10
            sen_mask = torch.tensor(sen_mask).cuda()

            if len(feature[0]) < 2:
                continue
            if not args.aspect_phrase:
                aspect.data.unsqueeze_(0)
            aspect.data.t_()
            target.data.sub_(1)

            ttarget = np.zeros((target.size(0), 2), dtype=int)
            for i, label in enumerate(target):
                if label.item() == 2:
                    continue
                elif label.item() == 3:
                    ttarget[i,0] = 1
                    ttarget[i,1] = 1
                else:
                    ttarget[i,label.item()] = 1
            ttarget = torch.tensor(ttarget, dtype=torch.float)

            if args.cuda:
                feature, aspect, ttarget = feature.cuda(), aspect.cuda(), ttarget.cuda()
            optimizer.zero_grad()
            logit, o_loss = model(feature, aspect, sen_mask)


            criterion = torch.nn.MultiLabelSoftMarginLoss()
            loss = criterion(logit, ttarget) + 1 * o_loss
            loss.backward()

            optimizer.step()

            steps += 1


            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)

        dev_acc, _= eval(dev_iter, model, args)
        if args.early > 0:
            if dev_acc > b_acc:
                early = 0
                b_acc = dev_acc
                torch.save(model.state_dict(), 'model/'+str(dev_acc)+'.pt')
            else:
                early += 1
                state_dict = torch.load('model/'+str(b_acc)+'.pt')
                model.load_state_dict(state_dict)
                model.cuda()
            if early >= args.early:
                lr_now *= 0.8
                if lr_now < 1e-4:
                    break
                early = 0
                optimizer = torch.optim.Adagrad(model.parameters(), lr=lr_now, weight_decay=args.l2, lr_decay=args.lr_decay)
        else:
            b_acc = dev_acc
    test_acc, _ = eval(test_iter, model, args, True)
    if args.verbose == 1:
        delta_time = time.time() - start_time
        print('\n{:.4f} - {:.4f}'.format(test_acc, delta_time))
        time_stamps.append((test_acc, delta_time))
        print()


    return test_acc, time_stamps


def eval(data_iter, model, args, conf=False):
    model.eval()
    corrects, avg_loss = 0, 0
    loss = None
    for batch in data_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment
        temp_m = feature.permute(1,0).data.numpy()
        sen_mask = np.empty(temp_m.shape, dtype=np.float32)
        sen_mask[temp_m!=1] = 0.
        sen_mask[temp_m==1] = -9.9e10
        sen_mask = torch.tensor(sen_mask).cuda()

        if not args.aspect_phrase:
            aspect.data.unsqueeze_(0)
        aspect.data.t_()
        target.data.sub_(1) 
        if args.cuda:
            feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

        logit, _ = model(feature, aspect, sen_mask)
        logit = torch.sigmoid(logit)

        t = target.cpu().data.numpy()
        predict = logit.cpu().data.numpy()
        predict = predict>0.5
        ppp = np.empty(t.shape, dtype=int)
        count = [0] * 4
        for i, sample in enumerate(predict):
            if sample[0] and sample[1]:
                ppp[i] = 3
                count[3] += 1
            elif sample[0]:
                ppp[i] = 0
                count[0] += 1
            elif sample[1]:
                ppp[i] = 1
                count[1] += 1
            else:
                ppp[i] = 2
                count[2] += 1

        correct = ppp == t
        corrects += correct.sum()
    if conf:
        confusion_matrix(ppp, t, args.class_num)
    res_dict = {'predict': list(predict), 'target': list(t), 'is_correct': list(correct)}
    size = len(data_iter.dataset)
    accuracy = corrects.item() * 100.0 / size
    model.train()

    return accuracy, res_dict 

def confusion_matrix(predict, target, class_num):
    cm = np.zeros((class_num, class_num), dtype=int)
    for i in range(class_num):
        unique, count = np.unique(predict[target==i], return_counts=True)
        count_dict = dict(zip(unique, count))
        for j in count_dict:
            cm[i,j] = count_dict[j]
    print('-'*10)
    print(cm)
    print('-'*10)
