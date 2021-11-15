from __future__ import print_function
import numpy as np

import os

import paddle

import PIL
import pickle

from paddle import fluid
from tqdm import tqdm
import time

from . import Algorithm
from pdb import set_trace as breakpoint


def accuracy(output, target, topk=(1,),flag='paddle'):
    """Computes the precision@k for the specified values of k"""
    if flag=='torch':
        maxk = max(topk)

        batch_size = target.shape[0]
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(correct)
        print('------')
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)

            res.append(correct_k.mul_(100.0 / batch_size))
        return np.array([ele.data.cpu().numpy() for ele in res])
    elif flag=='paddle':
        maxk = max(topk)
        batch_size = target.shape[0]
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target=paddle.reshape(target,(1, -1)).expand_as(pred)

        # print(pred.shape)
        # print(target.shape)
        correct = pred.equal(target)
        # print(correct)
        res = []
        for k in topk:
            correct_k = paddle.reshape(correct[:k],[-1]).numpy()
            correct_k=correct_k.sum(0)
            
            res.append((correct_k*100.0)/ batch_size)
        return np.array(res)
class ClassificationModel(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.networks['model']=self.networks['model']
    def allocate_tensors(self):
        self.tensors = {}
        # self.tensors['dataX'] = paddle.Tensor().astype('float32')
        # self.tensors['labels'] = paddle.Tensor().astype('int64')

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        # with fluid.dygraph.guard():
        start = time.time()
        self.tensors['dataX']=batch[0]
        self.tensors['labels']=batch[1]
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        
        
        

        batch_load_time = time.time() - start
        #********************************************************

        #********************************************************
        start = time.time()
        if do_train: # zero the gradients
            self.optimizers['model'].clear_grad()
        #********************************************************

        #***************** SET TORCH fluid.dygraph.to_variableS ******************
        # dataX_var = torch.autograd.fluid.dygraph.to_variable(dataX, volatile=(not do_train))
        # with fluid.dygraph.guard():
        #     dataX_var = fluid.dygraph.to_variable(dataX)
        #     labels_var = fluid.dygraph.to_variable(labels)
        dataX_var=dataX
        labels_var =labels
        #********************************************************

        #************ FORWARD THROUGH NET ***********************
        pred_var = self.networks['model'](dataX_var)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        record = {}
        loss_total = self.criterions['loss'](pred_var, labels_var)
        
        # pre=pred_var.data
        pre=pred_var
        label =labels

        # print(pre.shape)
        # print(label.shape)
        # print(accuracy(pre, label, topk=(1,))[0])
        record['prec1'] = accuracy(pre, label, topk=(1,))[0]
        #
        record['loss'] = float(loss_total.numpy()[0])

        #  record['loss'] = loss_total.data
        # record['prec1'] = accuracy(pre, label, topk=(1,))[0][0]
        # record['loss'] = loss_total.data[0]
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            # print(loss_total)
            # print(pre.shape)
            self.optimizers['model'].step()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        # print(record)
        return record
