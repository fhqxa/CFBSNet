import CFBSNet.main._init_paths
from CFBSNet.lib.net import Network
from CFBSNet.lib.config import cfg, update_config
from CFBSNet.lib.dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from CFBSNet.lib.core.evaluate import FusionMatrix


def parse_args():
    parser = argparse.ArgumentParser(description="BBN evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="E:\\PycharmProjects\\CFBSNet\\\configs\\cifar100.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def valid_model(dataLoader, model, cfg, device, num_classes):
    result_list = []
    pbar = tqdm(total=len(dataLoader))
    model.eval()
    top1_count, top2_count, top3_count, index, fusion_matrix, head, middle, tail = (
        [],
        [],
        [],
        0,
        FusionMatrix(num_classes),
        [],
        [],
        [],
    )

    func = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (image, image_labels, meta) in enumerate(dataLoader):
            image = image.to(device)
            output = model(image)
            result = func(output)
            _, top_k = result.topk(5, 1, True, True)
            score_result = result.cpu().numpy()
            fusion_matrix.update(score_result.argmax(axis=1), image_labels.numpy())
            topk_result = top_k.cpu().tolist()
            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                result_list.append(
                    {
                        "image_id": image_id,
                        "image_label": int(image_labels[i]),
                        "top_3": topk_result[i],
                    }
                )
                top1_count += [topk_result[i][0] == image_labels[i]]
                if image_labels[i]>=0 and image_labels[i]<=40:
                    head += [topk_result[i][0] == image_labels[i]]
                elif image_labels[i]>=41 and image_labels[i]<=80:
                    middle += [topk_result[i][0] == image_labels[i]]
                elif image_labels[i]>=81 and image_labels[i]<=99:
                    tail += [topk_result[i][0] == image_labels[i]]
                # top2_count += [image_labels[i] in topk_result[i][0:2]]
                # top3_count += [image_labels[i] in topk_result[i][0:3]]
                index += 1
            now_acc = np.sum(top1_count) / index
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)
    top1_acc = float(np.sum(top1_count) / len(top1_count))
    head_acc = float(np.sum(head) / len(head))
    middle_acc = float(np.sum(middle) / len(middle))
    tail_acc = float(np.sum(tail) / len(tail))
    # top2_acc = float(np.sum(top2_count) / len(top1_count))
    # top3_acc = float(np.sum(top3_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}% \n head:{:>5.2f}%  middle:{:>5.2f}% tail:{:>5.2f}%".format(
            top1_acc * 100, head_acc * 100, middle_acc * 100, tail_acc * 100
        )
    )
    pbar.close()


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    model = Network(cfg, mode="test", num_classes=num_classes)

    model_dir = "F:\\徐峻岩备份\\桌面\\徐峻岩\\300轮\\终极调参\\50\\super 180 230 160d\\output\\cifar100\\BBN.CIFAR100.res32.200epoch\\models"
    # model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    model.load_model(model_path)

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_model(testLoader, model, cfg, device, num_classes)
