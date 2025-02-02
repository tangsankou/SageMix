#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


from SageMix import SageMix
from data import ModelNet40, ScanObjectNN
from model import PointNet, DGCNN
from util import cal_loss, cal_loss_mix, IOStream
from saliency import SphereSaliency


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    if args.data == 'MN40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class=40
    elif args.data == 'SONN_easy':
        train_loader = DataLoader(ScanObjectNN(partition='train', num_points=args.num_points, ver="easy"), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="easy"), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15
    elif args.data == 'SONN_hard':
        train_loader = DataLoader(ScanObjectNN(partition='train', num_points=args.num_points, ver="hard"), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="hard"), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15
    
    
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, num_class).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, num_class).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    

    sagemix = SageMix(args, num_class)
    criterion = cal_loss_mix

    # spherealiency = SphereSaliency(args, num_class)


    best_test_acc = 0
    for epoch in range(args.epochs):

        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in tqdm(train_loader):
            # print("---------label:",label.shape)
            data, label = data.to(device), label.to(device).squeeze()
            
            # print("---label:",label.shape)
            batch_size = data.size()[0]
            # print("---------data:",data.shape)#(32,1024,3)
            ####################
            # generate augmented sample
            ####################
            """ model.eval()
            data_var = Variable(data.permute(0,2,1), requires_grad=True)
            # print("data_var:",data_var.shape)#(32,3,1024)
            logits = model(data_var)
            loss = cal_loss(logits, label, smoothing=False)
            loss.backward()
            opt.zero_grad()
            saliency = torch.sqrt(torch.mean(data_var.grad**2,1))# (B,N) """
            ############
            #generate augmented sample
            ############
            model.eval()
            data_var = Variable(data.permute(0,2,1), requires_grad=True)
            logits = model(data_var)
            loss = cal_loss(logits, label, smoothing=False)
            loss.backward()
            grad = data_var.grad.data#(32,3,1024)
            opt.zero_grad()

            # print("grad:",grad.shape)
            # Change gradients into spherical axis and compute r*dL/dr
            sphere_core = torch.median(data, dim=1, keepdim=True)[0]#(32,1,1024)
            # print("sphere_core:",sphere_core.shape)
            sphere_r = torch.sqrt(torch.sum(torch.square(data - sphere_core), dim=2))  # BxN(32,1024)
            # print("sphere_r:",sphere_r.shape)
            sphere_axis = data - sphere_core  # BxNx3(32,1024,3)
            # print("sphere_axis:",sphere_axis.shape)

            sphere_map = torch.mul(torch.sum(torch.mul(grad.permute(0,2,1), sphere_axis), dim=2), torch.pow(sphere_r, args.power))
            # print("sphere_map:",sphere_map.shape)
            # saliency_map = spherealiency.compute_saliency(data)  # Compute saliency map
            saliency = sphere_map.to(device)  # Convert saliency map to torch tensor
            #end

            # print("saliency:",saliency.shape)
            data, label = sagemix.mix(data, label, saliency)
            model.train()
                
            opt.zero_grad()
            # print("data2:",data.shape)#(32,1024,3)
            logits = model(data.permute(0,2,1))
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            
        scheduler.step()
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss*1.0/count)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = cal_loss(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best test acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc,
                                                                              best_test_acc)
        io.cprint(outstr)
       


def test(args, io):
    if args.data == 'MN40':
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class=40
    elif args.data == 'SONN_easy':
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="easy"), 
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15
    elif args.data == 'SONN_hard':
        test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points, ver="hard"), 
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class =15
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, num_class).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, num_class).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in tqdm(test_loader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--data', type=str, default='MN40', metavar='N',
                        choices=['MN40', 'SONN_easy', 'SONN_hard']) #SONN_easy : OBJ_ONLY, SONN_hard : PB_T50_RS
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    
    parser.add_argument('--sigma', type=float, default=-1) 
    parser.add_argument('--theta', type=float, default=0.2) 
   
    parser.add_argument('--power', type=int, default=6, help='x: -dL/dr*r^x')

    args = parser.parse_args()

    if args.sigma==-1:
        if args.model=='dgcnn':
            args.sigma=0.3
        elif args.model=="pointnet":
            args.sigma=2.0
    
    if args.model=='dgcnn':
        args.use_sgd=True
    elif args.model=="pointnet":
        args.use_sgd=False

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
