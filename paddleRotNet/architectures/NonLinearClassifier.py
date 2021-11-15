import math
import numpy as np
import paddle
from paddle import nn, normal, reshape
from paddle.nn.functional import normalize


class BasicBlock(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.layers = nn.Sequential()
        self.layers.add_sublayer('Conv', nn.Conv2D(in_planes, out_planes,
                                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                                   bias_attr=False))
        self.layers.add_sublayer('BatchNorm', nn.BatchNorm2D(out_planes))
        self.layers.add_sublayer('ReLU', nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class GlobalAvgPool(nn.Layer):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, feat):
        assert (feat.shape[2] == feat.shape[3])
        feat_avg = reshape(nn.AvgPool2D(feat.shape[2],exclusive=False)(feat), (-1, feat.shape[1]))
        return feat_avg


class Flatten(nn.Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class Classifier(nn.Layer):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        nChannels = opt['nChannels']
        num_classes = opt['num_classes']
        self.cls_type = opt['cls_type']

        self.classifier = nn.Sequential()

        if self.cls_type == 'MultLayer':
            nFeats = min(num_classes * 20, 2048)
            self.classifier.add_sublayer('Flatten', Flatten())
            self.classifier.add_sublayer('Liniear_1', nn.Linear(nChannels, nFeats, bias_attr=None))
            self.classifier.add_sublayer('BatchNorm_1', nn.BatchNorm2D(nFeats))
            self.classifier.add_sublayer('ReLU_1', nn.ReLU())
            self.classifier.add_sublayer('Liniear_2', nn.Linear(nFeats, nFeats, bias_attr=None))
            self.classifier.add_sublayer('BatchNorm2D', nn.BatchNorm2D(nFeats))
            self.classifier.add_sublayer('ReLU_2', nn.ReLU())
            self.classifier.add_sublayer('Liniear_F', nn.Linear(nFeats, num_classes))
        elif self.cls_type == 'NIN_ConvBlock3':
            self.classifier.add_sublayer('Block3_ConvB1', BasicBlock(nChannels, 192, 3))
            self.classifier.add_sublayer('Block3_ConvB2', BasicBlock(192, 192, 1))
            self.classifier.add_sublayer('Block3_ConvB3', BasicBlock(192, 192, 1))
            self.classifier.add_sublayer('GlobalAvgPool', GlobalAvgPool())
            self.classifier.add_sublayer('Liniear_F', nn.Linear(192, num_classes))
        elif self.cls_type == 'Alexnet_conv5' or self.cls_type == 'Alexnet_conv4':
            if self.cls_type == 'Alexnet_conv4':
                block5 = nn.Sequential(
                    nn.Conv2D(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2D(256),
                    nn.ReLU(),
                )
                self.classifier.add_sublayer('ConvB5', block5)
            self.classifier.add_sublayer('Pool5', nn.MaxPool2D(kernel_size=3, stride=2))
            self.classifier.add_sublayer('Flatten', Flatten())
            self.classifier.add_sublayer('Linear1', nn.Linear(256 * 6 * 6, 4096, bias=False))
            self.classifier.add_sublayer('BatchNorm1', nn.BatchNorm1d(4096))
            self.classifier.add_sublayer('ReLU1', nn.ReLU())
            self.classifier.add_sublayer('Liniear2', nn.Linear(4096, 4096, bias=False))
            self.classifier.add_sublayer('BatchNorm2', nn.BatchNorm1d(4096))
            self.classifier.add_sublayer('ReLU2', nn.ReLU())
            self.classifier.add_sublayer('LinearF', nn.Linear(4096, num_classes))
        else:
            raise ValueError('Not recognized classifier type: %s' % self.cls_type)

        self.initilize()

    def forward(self, feat):
        return self.classifier(feat)

    def initilize(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight.set_value(paddle.normal(shape=m.weight.shape, mean=0, std=math.sqrt(2. / n)))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.full_like(m.weight, fill_value=1))
                m.bias.set_value(paddle.zeros_like(m.bias))
            elif isinstance(m, nn.Linear):
                fin = m.weight.shape[0]
                fout = m.weight.shape[1]
                std_val = np.sqrt(2.0 / fout)
                m.weight.set_value(paddle.normal(shape=m.weight.shape, mean=0, std=std_val))
                if m.bias is not None:
                    m.bias.set_value(paddle.zeros_like(m.bias))


def create_model(opt):
    return Classifier(opt)
