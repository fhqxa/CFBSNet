from CFBSNet.lib.dataset.baseset import BaseSet
from torch.utils.data import Dataset
import random, cv2
import numpy as np, torch, torch.nn as nn, os
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from PIL import Image
from CFBSNet.lib.dataset import comparison0, dataset0, label0, utility0

import torchvision


class sundata(Dataset):
    cls_num = 324

    def __init__(self, mode, cfg, transform=None, target_transform=None):
        train = True if mode == "train" else False
        self.cfg = cfg
        self.train = train
        self.dual_sample = True if cfg.TRAIN.SAMPLER.DUAL_SAMPLER.ENABLE and self.train else False
        rand_number = cfg.DATASET.IMBALANCECIFAR.RANDOM_SEED
        if self.train:
            torch.manual_seed(rand_number)  # cpu
            torch.cuda.manual_seed(rand_number)  # gpu
            torch.cuda.manual_seed_all(rand_number)  # multi-gpu
            np.random.seed(rand_number)  # numpy
            random.seed(rand_number)  # random and transforms
            torch.backends.cudnn.deterministic = True  # cudnn
            torch.backends.cudnn.benchmark = False

            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        path_all_dataset = 'F:\\Datasets\\'
        path_all_dataset_usb = 'F:\\Datasets\\'
        # switch_dataset = 'SUN397'
        # switch_dataset = 'VOC2012_Per'
        switch_dataset = 'VOC2012_PerBir'
        switch_imbalance = 'Original'
        image_path, image_path_test, transform, transform_test, num_cls, num_cls_c_semantic, relation_semantic, \
        eval_tree = \
            dataset0.AllDatasets(switch_dataset, path_all_dataset, path_all_dataset_usb, switch_imbalance)
        self.image_path_test=image_path_test
        self.transform_test=transform_test
        self.image_path=image_path
        self.transform=transform


        train_dataset = dataset0.DatasetFromPath(image_path, transform)
        test_dataset = dataset0.DatasetFromPath(image_path_test, transform_test)

        train_labels, test_labels = [i[1] for i in image_path], [i[1] for i in image_path_test]

        _, percls, percls_test, _ = dataset0.NumPerclass(train_labels, test_labels, switch_imbalance)
        percls_c=dataset0.NumPerclass_Coarse(num_cls,num_cls_c_semantic,relation_semantic,percls)

        self.percls_c=percls_c
        self.ftoc=relation_semantic

        self.train_dataset=train_dataset


        self.test_dataset=test_dataset

        self.train_labels=train_labels

        self.test_labels=test_labels

        self.percls=percls

        self.percls_test=percls_test
        # print('666666666666')
        # print(train_dataset)
        # print('66666666666666')

        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train):

            self.class_weight, self.sum_weight = self.get_weight()

            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        # img, target,label_c = self.image_path[index]

        meta = dict()
        if self.train:
            img, target, label_c = self.image_path[index]
            img = Image.open(img).convert('RGB')
            img = self.transform(img)
        else:

            img, target, label_c = self.image_path_test[index]
            img = Image.open(img).convert('RGB')
            img = self.transform_test(img)
        # img = Image.fromarray(img)
        # img = Image.open(img).convert('RGB')
        # img, target=dataset0.DatasetFromPath(self.image_path[index], self.transform)
        if self.dual_sample:
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            # sample_img, sample_label = dataset0.DatasetFromPath(self.image_path[sample_index], self.transform)
            sample_img, sample_label, label_c = self.image_path[sample_index]
            sample_img = Image.open(sample_img).convert('RGB')
            sample_img = self.transform(sample_img)
            # sample_img=list(sample_img)
            # sample_img = [item.cpu().numpy() for item in sample_img]
            # sample_img=np.array(sample_img)
            # sample_img = Image.fromarray(sample_img)
            # img = Image.open(img).convert('RGB')
            # sample_img = self.transform(sample_img)

            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label
        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # img=np.array(img)
        # target=np.array(target)
        # meta=np.array(meta)
        return img, target, meta

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def get_weight(self):
        max_num = max(self.percls)
        class_weight = [max_num / i for i in self.percls]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.train_labels):
            cat_id = anno
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_traindataset(self):
        return self.train_dataset

    def get_trarinlabel(self):
        return self.train_labels

    def get_testdataset(self):
        return self.test_dataset

    def get_testlebaels(self):
        return self.test_labels

    def get_ftoc(self):
        return self.ftoc

    def get_num_classes(self):
        return self.cls_num

    def get_per_cls_train(self):
        return self.percls

    def __len__(self):
        if self.train:
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)

    def get_per_cls_test(self):
        return self.percls_test

    def get_percls_c(self):
        return self.percls_c

