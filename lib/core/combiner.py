import numpy as np
import torch, math
from CFBSNet.lib.core.evaluate import accuracy

class Combiner:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.type = cfg.TRAIN.COMBINER.TYPE
        self.device = device
        self.epoch_number = cfg.TRAIN.MAX_EPOCH
        self.func = torch.nn.Softmax(dim=1)
        self.initilize_all_parameters()

    def initilize_all_parameters(self):
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number

    def get_lambda(self, cur_epoch, coarse_train_ep, fine_train_ep):

        if cur_epoch > self.epoch_number - fine_train_ep - 1:
            my_lambda = 0

        elif cur_epoch < coarse_train_ep:
            my_lambda = 1

        else:
            my_lambda = 1 - ((cur_epoch - coarse_train_ep) / (self.epoch_number - coarse_train_ep - fine_train_ep - 1)) ** 2

        return my_lambda

    def get_scalingfac(self, num1, num2):
        s1 = int(math.floor(math.log10(num1)))
        s2 = int(math.floor(math.log10(num2)))
        scale = 10 ** (s1 - s2)
        return scale

    def reset_epoch(self, epoch):
        self.epoch = epoch
    

    def forward(self, model, criterion, image, label, meta, coarse_label,**kwargs):
        return eval("self.{}".format(self.type))(
            model, criterion, image, label, meta, coarse_label,**kwargs
        )

    def default(self, model, criterion, image, label, **kwargs):
        image, label = image.to(self.device), label.to(self.device)
        output = model(image)
        loss = criterion(output, label)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        return loss, now_acc


    def bbn_mix(self, model, criterion, image, label, meta, coarse_label, **kwargs):

        image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)
        label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)

        feature_a, feature_b = (
            model(image_a, feature_cb=True),
            model(image_b, feature_rb=True),
        )

        coarse_label = coarse_label.long().to(self.device)

        output_a=model(feature_a,classifier_flag=True)
        output_b = model(feature_b, classifier_coarse=True)

        my_lambda = self.get_lambda(self.epoch, coarse_train_ep=50, fine_train_ep=200)
        if my_lambda == 0:
            loss = criterion(output_a, label_a).to(self.device)
        elif my_lambda == 1:
            loss = criterion(output_b, coarse_label).to(self.device)
        else:
            loss1 = criterion(output_b, coarse_label).to(self.device)
            loss2 = criterion(output_a, label_a).to(self.device)
            scale = self.get_scalingfac(loss1, loss2)
            loss = my_lambda * loss1 + (1 - my_lambda) * scale * loss2
###
        now_result_a = torch.argmax(self.func(output_a), 1)
        now_result_b = torch.argmax(self.func(output_b), 1)

        now_acc_a = accuracy(now_result_a.cpu().numpy(), label_a.cpu().numpy())[0]
        now_acc_b = accuracy(now_result_b.cpu().numpy(), label_a.cpu().numpy())[0]

        return loss, now_acc_a

