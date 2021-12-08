#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###
# Adapted to petastorm dataset using original contents from
# https://github.com/pytorch/examples/mnist/main.py .
###
from __future__ import division, print_function

import argparse

# Must import pyarrow before torch. See: https://github.com/uber/petastorm/blob/master/docs/troubleshoot.rst
import pyarrow  # noqa: F401 pylint: disable=W0611
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from examples.mnist import DEFAULT_MNIST_DATA_PATH
from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader

import pytorch_lightning as pl
from torchmetrics import Accuracy, Metric
from typing import Dict, Any, Union, List, Optional, Union, Callable, AnyStr
from pytorch_lightning.loggers import TensorBoardLogger

import torch.utils.data as d

from torchvision.datasets import MNIST
from torch.utils.data import Dataset

from PIL import Image


class Support(Metric):
    def __init__(self,  n_classes: int = 10,
                        compute_on_step: bool = True,
                        dist_sync_on_step: bool = False,
                        process_group: Optional[Any] = None,
                        dist_sync_fn: Callable = None) -> None:

        super().__init__(compute_on_step=compute_on_step,
                        dist_sync_on_step=dist_sync_on_step,
                        process_group=process_group,
                        dist_sync_fn=dist_sync_fn)

        self.n_classes = n_classes
        self.add_state("counts", default = torch.zeros(self.n_classes),
                                dist_reduce_fx="sum")

    def update(self, _preds: torch.Tensor, target: torch.Tensor) -> None:
        values = torch.bincount(target, minlength=self.n_classes)
        self.counts += values

    def compute(self) -> Dict[AnyStr,torch.Tensor]:
        return {str(i):self.counts[i].item() for i in range(self.n_classes)} #TODO does str work here?

class Net(pl.LightningModule):
    def __init__(self, momentum, lr):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.momentum = momentum
        self.lr = lr

        self.support_val = Support()
        self.support_tr = Support()


    # pylint: disable=arguments-differ
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        data, target = batch['image'], batch['digit']
        loss = F.nll_loss(self(data), target)
        self.log('loss_train', loss)

        output = self(data)
        self.support_tr.update(_preds= output, target = batch['digit'])

        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        print('\n Training class frequencies: \n', self.support_tr.compute())
        self.support_tr.reset()


    def validation_step(self, batch, batch_idx):
        data, target = batch['image'], batch['digit']
        output = self(data)
        loss = F.nll_loss(output, target, reduction='sum')  # sum up batch loss
        self.support_val.update(_preds= output, target = batch['digit'])
        #self.log('support', self.support_val, on_step = False, on_epoch = True)

        return loss

    def on_validation_epoch_end(self) -> None:
        print('\n Validation class frequencies: \n', self.support_val.compute())
        self.support_val.reset()


    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)


def _transform_row(mnist_row):
    # For this example, the images are stored as simpler ndarray (28,28), but the
    # training network expects 3-dim images, hence the additional lambda transform.
    transform = transforms.Compose([
        transforms.Lambda(lambda nd: nd.reshape(28, 28, 1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # In addition, the petastorm pytorch DataLoader does not distinguish the notion of
    # data or target transform, but that actually gives the user more flexibility
    # to make the desired partial transform, as shown here.
    result_row = {
        'image': transform(mnist_row['image']),
        'digit': mnist_row['digit']
    }

    return result_row

class TorchMNIST(MNIST):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        """
        img, target = self.data[index], int(self.targets[index]) #img is Tensor

        img = Image.fromarray(img.numpy(), mode="L")
        
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        
        return {'image': transform(img), 'digit':target}

def main():

     # Training settings
    parser = argparse.ArgumentParser(description='Petastorm MNIST Example')
    default_dataset_url = 'file://{}'.format(DEFAULT_MNIST_DATA_PATH)
    parser.add_argument('--dataset-url', type=str,
                        default=default_dataset_url, metavar='S',
                        help='hdfs:// or file:/// URL to the MNIST petastorm dataset '
                             '(default: %s)' % default_dataset_url)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--do_eval', action='store_true', default=True,
                        help='perform validation step after each training step?')
    parser.add_argument('--use_torch_loader', action='store_false', default=True,
                        help='Make use of the petastorm DataLoader in place of the pytorch DataLoader')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpus', default=1, #TODO need to debug multi-gpu
                        help='Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    model =  Net(lr=args.lr, momentum=args.momentum)
    logger = TensorBoardLogger("tb_logs", name="my_model")
    transform = TransformSpec(_transform_row, removed_fields=['idx'])

    if args.use_torch_loader:
        if args.do_eval:
            trainer = pl.Trainer(check_val_every_n_epoch=1, gpus = args.gpus, num_sanity_val_steps=0, max_epochs = args.epochs, logger = logger) 
            with DataLoader(make_reader('{}/train'.format(args.dataset_url), 
                                        #num_epochs=args.epochs,
                                        transform_spec=transform),
                            batch_size=args.batch_size) as train_dataset:
                with DataLoader(make_reader('{}/test'.format(args.dataset_url), 
                                            #num_epochs=args.epochs,
                                            transform_spec=transform),
                                batch_size=args.test_batch_size) as eval_dataset:
                    trainer.fit(model,train_dataset,eval_dataset)
        else:
            trainer = pl.Trainer(check_val_every_n_epoch=0,  gpus = args.gpus, num_sanity_val_steps=0, max_epochs = args.epochs, logger = logger)
            with DataLoader(make_reader('{}/train'.format(args.dataset_url), 
                                        #num_epochs=args.epochs,
                                        transform_spec=transform),
                            batch_size=args.batch_size) as train_dataset:
                    trainer.fit(model, train_dataset, DataLoader([["dummy"]]))


    else: #use pytorch dataloader
        print ('Using Pytorch Dataloader')
        if args.do_eval:
            trainer = pl.Trainer(check_val_every_n_epoch=1, gpus = args.gpus, num_sanity_val_steps=0, max_epochs = args.epochs, logger = logger) 
            train_dataset = d.DataLoader(TorchMNIST(root = '/projects/bdata/trelium/MNIST_torch', download=True, train=True), batch_size=args.batch_size) 
            eval_dataset = d.DataLoader(TorchMNIST(root = '/projects/bdata/trelium/MNIST_torch', download=True, train=False),batch_size=args.test_batch_size)  
            trainer.fit(model,train_dataset,eval_dataset)
        else:
            trainer = pl.Trainer(check_val_every_n_epoch=0,  gpus = args.gpus, num_sanity_val_steps=0, max_epochs = args.epochs, logger = logger)
            train_dataset = d.DataLoader(TorchMNIST(root = '/projects/bdata/trelium/MNIST_torch', download=True, train=True), batch_size=args.batch_size) 

            trainer.fit(model, train_dataset, DataLoader([["dummy"]]))




if __name__ == '__main__':
    main()


