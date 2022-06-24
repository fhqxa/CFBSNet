import CFBSNet.main._init_paths
import sys
import numpy as np
from CFBSNet.lib.loss import *
from CFBSNet.lib.dataset import *
from CFBSNet.lib.dataset import sundata
from CFBSNet.lib.config import cfg, update_config
from CFBSNet.lib.utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_category_list,
)
from CFBSNet.lib.core.function import train_model, valid_model
from CFBSNet.lib.core.combiner import Combiner
from CFBSNet.lib.dataset.sundata import *
import torch
import os, shutil
from torch.utils.data import DataLoader
import argparse
import warnings
import click
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import ast


def parse_args():
    parser = argparse.ArgumentParser(description="codes for BBN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="E:\\PycharmProjects\\BBN\\\configs\\cifar100.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type= ast.literal_eval,
        dest = 'auto_resume',
        required=False,
        default= True,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    logger, log_file = create_logger(cfg)
    warnings.filterwarnings("ignore")
    cudnn.benchmark = True
    auto_resume = args.auto_resume


    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    coarse_label = train_set.get_coarse()
    valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)


    annotations = train_set.get_annotations()
    coarse_idx = []
    for i in range(100):
        for j,x in zip(annotations,coarse_label):
            if j['category_id'] == i:
                coarse_idx.append(x['coarse_id'])
                break

    num_classes = train_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")

    num_class_list, cat_list = get_category_list(annotations, num_classes, cfg)
    coarse_idx = np.array(coarse_idx)

    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfg": cfg,
        "device": device,
    }



    criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)

    epoch_number = cfg.TRAIN.MAX_EPOCH

    # ----- BEGIN MODEL BUILDER -----
    model = get_model(cfg, num_classes, device, logger)
    combiner = Combiner(cfg, device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    # ----- END MODEL BUILDER -----



    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )

    validLoader = DataLoader(
        valid_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # close loop
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes")
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard")
        if cfg.TRAIN.TENSORBOARD.ENABLE
        else None
    )

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        logger.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        )
        if not click.confirm(
            "\033[1;31;40mContinue and override the former directory?\033[0m",
            default=False,
        ):
            exit(0)
        shutil.rmtree(code_dir)
        if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
    print("=> output model will be saved in {}".format(model_dir))
    this_dir = os.path.dirname(__file__)
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
    )
    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)

    if tensorboard_dir is not None:
        dummy_input = torch.rand((1, 3) + cfg.INPUT_SIZE).to(device)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_graph(model if cfg.CPU_MODE else model.module, (dummy_input,))
    else:
        writer = None

    best_result, best_epoch, start_epoch = 0, 0, 1
    # ----- BEGIN RESUME ---------
    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume == False:
        auto_resume = False
    else:
        all_models.remove("best_model.pth")
        resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir, "epoch_{}.pth".format(resume_epoch))

    if cfg.RESUME_MODEL != "" or auto_resume:
        if cfg.RESUME_MODEL == "":
            resume_model = resume_model_path
        else:
            resume_model = cfg.RESUME_MODEL if '/' in cfg.RESUME_MODEL else os.path.join(model_dir, cfg.RESUME_MODEL)
        logger.info("Loading checkpoint from {}...".format(resume_model))
        checkpoint = torch.load(
            resume_model, map_location="cpu" if cfg.CPU_MODE else "cuda"
        )
        if cfg.CPU_MODE:
            model.load_model(resume_model)
        else:
            model.module.load_model(resume_model)
        if cfg.RESUME_MODE != "state_dict":
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint['best_result']
            best_epoch = checkpoint['best_epoch']
    # ----- END RESUME ---------

    logger.info(
        "-------------------Train start :{}  {}  {}-------------------".format(
            cfg.BACKBONE.TYPE, cfg.MODULE.TYPE, cfg.TRAIN.COMBINER.TYPE
        )
    )

    for epoch in range(start_epoch, epoch_number + 1):
        scheduler.step()

        # # 'None'
        # train_sampler = None
        # per_cls_weights = None
        # # 'Resample':
        # train_sampler = ImbalancedDatasetSampler(train_dataset)
        # per_cls_weights = None
        # # 'Reweight':
        # train_sampler = None
        # beta = 0.9999
        # effective_num = 1.0 - np.power(beta, num_class_list)
        # per_cls_weights = (1.0 - beta) / np.array(effective_num)
        # per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(num_class_list)
        # per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        # 'DRW':
        # train_sampler = None
        # idx = epoch// 240
        # betas = [0, 0.9999]
        # effective_num = 1.0 - np.power(betas[idx], num_class_list)
        # per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        # per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(num_class_list)
        # per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        # #coarse drw
        # train_sampler = None
        # idx = epoch // 240
        # betas = [0, 0.9999]
        # effective_num = 1.0 - np.power(betas[idx], per_cls_c)
        # c_per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        # c_per_cls_weights = c_per_cls_weights / np.sum(c_per_cls_weights) * len(per_cls_c)
        # c_per_cls_weights = torch.FloatTensor(c_per_cls_weights).to(device)

        # criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)
        # criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict,weight=per_cls_weights)
        # criterion = eval(cfg.LOSS.LOSS_TYPE)(cls_num_list=num_class_list,device=device, weight=None)
        # criterion = eval(cfg.LOSS.LOSS_TYPE)(gamma=1, weight=None)
        # criterion_c = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict,weight=c_per_cls_weights)

        train_acc, train_loss = train_model(
            trainLoader,
            model,
            epoch,
            epoch_number,
            optimizer,
            combiner,
            criterion,
            cfg,
            logger,
            coarse_idx,
            writer=writer,
        )
        model_save_path = os.path.join(
            model_dir,
            "epoch_{}.pth".format(epoch),
        )
        if epoch % cfg.SAVE_STEP == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_result': best_result,
                'best_epoch': best_epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_save_path)

        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            valid_acc, valid_loss = valid_model(
                validLoader, epoch, model, cfg, criterion, logger, device, writer=writer
            )
            loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc
            if valid_acc > best_result:
                best_result, best_epoch = valid_acc, epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model.pth")
                )
            logger.info(
                "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                    best_epoch, best_result * 100
                )
            )
        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()
    logger.info(
        "-------------------Train Finished :{}-------------------".format(cfg.NAME)
    )
