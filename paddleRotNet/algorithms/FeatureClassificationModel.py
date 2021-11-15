import numpy as np
import paddle,torch

import time

from . import Algorithm
from pdb import set_trace as breakpoint

# from paddleRotNet.algorithms import Algorithm
# SEED=100
# torch.manual_seed(SEED)
# paddle.seed(SEED)
# np.random.seed(SEED)

def accuracy(output, target, topk=(1,),flag='paddle'):
    """Computes the precision@k for the specified values of k"""
    if flag=='torch':
        maxk = max(topk)

        batch_size = target.shape[0]
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(correct)
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
class FeatureClassificationModel(Algorithm):
    def __init__(self, opt):
        self.out_feat_keys = opt['out_feat_keys']
        Algorithm.__init__(self, opt)

    def allocate_tensors(self):
        self.tensors = {}
        # self.tensors['dataX'] = paddle.to_tensor(dtype='float32')
        # self.tensors['labels'] = paddle.to_tensor(dtype='int64')

    def train_step(self, batch):
        # print(batch[0].shape)
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        # print('0---------')
        # *************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        # self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        # self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        self.tensors['dataX'] = batch[0]
        self.tensors['labels'] = batch[1]
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        batch_load_time = time.time() - start
        # ********************************************************

        # ********************************************************
        start = time.time()
        out_feat_keys = self.out_feat_keys
        finetune_feat_extractor=None
        try:
            finetune_feat_extractor = self.optimizers['feat_extractor'] is not None
        except:
            # print('not finetune_feat_extractor ')
            pass

        if do_train:  # zero the gradients
            self.optimizers['classifier'].clear_grad()
            if finetune_feat_extractor:
                self.optimizers['feat_extractor'].clear_grad()
            else:
                self.networks['feat_extractor'].eval()
        # ********************************************************

        # ***************** SET TORCH fluid.dygraph.to_variableS ******************
        import paddle.fluid as fluid
        import numpy as np
        # with fluid.dygraph.guard():
        dataX_var = dataX
            # dataX_var = fluid.dygraph.to_variable(dataX, volatile=((not do_train) or (not finetune_feat_extractor)))
        labels_var = labels
            # labels_var = fluid.dygraph.to_fluid.dygraph.to_variable(labels, requires_grad=False)
        # ********************************************************

        # ************ FORWARD PROPAGATION ***********************
        # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh', )
        # print(dataX_var)
        feat_var = self.networks['feat_extractor'](dataX_var, out_feat_keys=out_feat_keys)
        if not finetune_feat_extractor:
            if isinstance(feat_var, (list, tuple)):
                for i in range(len(feat_var)):
                    feat_var[i] = fluid.dygraph.to_variable(feat_var[i].numpy())
            else:
                feat_var = fluid.dygraph.to_variable(feat_var.numpy())
        pred_var = self.networks['classifier'](feat_var)
        # ********************************************************

        # *************** COMPUTE LOSSES *************************
        record = {}
        if isinstance(pred_var, (list, tuple)):
            loss_total = None
            for i in range(len(pred_var)):
                loss_this = self.criterions['loss'](pred_var[i], labels_var)
                loss_total = loss_this if (loss_total is None) else (loss_total + loss_this)
                record['prec1_c' + str(1 + i)] = accuracy(pred_var[i].numpy(), labels, topk=(1,))[0][0]
                record['prec5_c' + str(1 + i)] = accuracy(pred_var[i].numpy(), labels, topk=(5,))[0][0]
        else:
            loss_total = self.criterions['loss'](pred_var, labels_var)
            record['prec1'] = accuracy(pred_var, labels, topk=(1,))[0]
            record['prec5'] = accuracy(pred_var, labels, topk=(5,))[0]
        record['loss'] = float(loss_total.numpy()[0])
        # ********************************************************

        # ****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['classifier'].step()
            if finetune_feat_extractor:
                self.optimizers['feat_extractor'].step()
        # ********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100 * (batch_load_time / total_time)
        record['process_time'] = 100 * (batch_process_time / total_time)

        return record
