import argparse
import datetime
import json
import logging
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# import wandb
from scipy import stats
from timm.utils import AverageMeter  # accuracy
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from IQA import IQA_build_loader
from config import get_config
from logger import create_logger
from lr_scheduler import build_scheduler
from models.build import build_model
# from models.build import build_model_t

from optimizer import build_optimizer
from utils import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    load_checkpoint,
    load_pretrained,
    reduce_tensor,
    save_checkpoint,
    update_teacher_model,
    copy_main_model1
)


# from Saliency.src.model import SODModel

def parse_option():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
             "full: cache all data, "
             "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Using tensorboard to track the process",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used (deprecated!)",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use torchinfo to show the flow of tensor in model",
    )
    parser.add_argument(
        "--repeat", action="store_true", help="Test model for publications"
    )
    parser.add_argument("--rnum", type=int, help="Repeat num")
    # distributed training
    # os.environ["LOCAL_RANK"] = '0'
    # os.environ["RANK"] = '0'
    # os.environ["WORLD_SIZE"]='1'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'

    local_rank = int(os.environ["LOCAL_RANK"])
    args, unparsed = parser.parse_known_args()
    os.environ['WANDB_MODE'] = 'offline'

    config = get_config(args, local_rank)
    return args, config


def main(config):

    wandb_runner = None

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    # t_model = build_model_t(config)

    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    # t_model.cuda()
    # t_model.eval()
    model_without_ddp = model

    if config.DEBUG_MODE:
        summary(
            model,
            (config.DATA.BATCH_SIZE, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
        )
        return
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = IQA_build_loader(config)
    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    # t_model = torch.nn.parallel.DistributedDataParallel(
    #     t_model,
    #     device_ids=[config.LOCAL_RANK],
    #     broadcast_buffers=False,
    #     find_unused_parameters=True,
    # )

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = torch.nn.SmoothL1Loss()

    max_plcc = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        max_plcc, epochs = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger
        )
        srcc, plcc, loss = validate(
            config, data_loader_val, model, epochs, len(dataset_val)
        )
        logger.info(
            f"SRCC and PLCC of the network on the {len(dataset_val)} test images: {srcc:.6f}, {plcc:.6f}"
        )
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        srcc, plcc, loss = validate(config, data_loader_val, model)
        logger.info(
            f"SRCC and PLCC of the network on the {len(dataset_val)} test images: {srcc:.6f}, {plcc:.6f}"
        )

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    if config.TENSORBOARD:
        writer = SummaryWriter(log_dir=config.OUTPUT)
    else:
        writer = None

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        srcc, plcc, loss = validate(
            config,
            data_loader_val,
            model,
            # model_saliency,
            epoch,
            len(dataset_val),
            tensorboard=writer,
            wandb_runner=wandb_runner,
        )
        if config.TENSORBOARD == True:
            writer.add_scalars(
                "Validate Metrics",
                {"SRCC": srcc, "PLCC": plcc},
                epoch,
            )

        logger.info(
            f"SRCC and PLCC of the network on the {len(dataset_val)} test images: {srcc:.6f}, {plcc:.6f}"
        )
        if plcc >= max_plcc:
            max_plcc = max(max_plcc, plcc)
            max_srcc = srcc
        elif plcc < 0:
            max_srcc = 0
        logger.info(f"Max PLCC: {max_plcc:.6f} Max SRCC: {max_srcc:.6f}")
        # if dist.get_rank() == 0:
        #     wandb_runner.summary["Best PLCC"], wandb_runner.summary["Best SRCC"] = (
        #         max_plcc,
        #         max_srcc,
        #     )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    writer.close()
    if dist.get_rank() == 0:
        logging.shutdown()
    else:
        logging.shutdown()
    dist.barrier()
    return


@torch.no_grad()
def validate(
        config, data_loader, model,
        # model_saliency,
        epoch, val_len, tensorboard=None, wandb_runner=None
):
    criterion = torch.nn.SmoothL1Loss()
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    temp_pred_scores = []
    temp_gt_scores = []
    end = time.time()
    for idx, (images, _, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # output, _ = model(images)
            # pred_masks, _ = model_saliency(images)
            output, _ = model(images, _, _)
        target.unsqueeze_(dim=-1)
        temp_pred_scores.append(output.view(-1))
        temp_gt_scores.append(target.view(-1))
        # measure accuracy and record loss
        loss = criterion(output, target)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            if config.TENSORBOARD == True and tensorboard != None:
                tensorboard.add_scalar(
                    "Validating Loss",
                    loss_meter.val,
                    epoch * len(data_loader) + idx,
                )
            if wandb_runner:
                wandb_runner.log(
                    {
                        "Validating Loss": loss_meter.val,
                        "Validate Batch": epoch * len(data_loader) + idx,
                    }
                )
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    pred_scores = torch.cat(temp_pred_scores)
    gt_scores = torch.cat(temp_gt_scores)
    # For distributed parallel, collect all data and then run metrics.
    if torch.distributed.is_initialized():
        preds_gather_list = [
            torch.zeros_like(pred_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(preds_gather_list, pred_scores)
        gather_preds = torch.cat(preds_gather_list, dim=0)[:val_len]
        gather_preds = (
            (gather_preds.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        grotruth_gather_list = [
            torch.zeros_like(gt_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(grotruth_gather_list, gt_scores)
        gather_grotruth = torch.cat(grotruth_gather_list, dim=0)[:val_len]
        gather_grotruth = (
            (gather_grotruth.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        final_preds = gather_preds.cpu().tolist()
        final_grotruth = gather_grotruth.cpu().tolist()

    test_srcc, _ = stats.spearmanr(final_preds, final_grotruth)
    test_plcc, _ = stats.pearsonr(final_preds, final_grotruth)
    logger.info(f" * SRCC@ {test_srcc:.6f} PLCC@ {test_plcc:.6f}")
    return test_srcc, test_plcc, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        return


if __name__ == "__main__":
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    dist.barrier()

    if args.repeat:
        assert args.rnum > 1
        num = args.rnum
    else:
        num = 1
    base_path = config.OUTPUT
    logger = logging.getLogger(name=f"{config.MODEL.NAME}")
    for i in range(num):

        if num > 1:
            config.defrost()
            config.OUTPUT = os.path.join(base_path, str(i))
            config.EXP_INDEX = i + 1
            config.SET.TRAIN_INDEX = None
            config.SET.TEST_INDEX = None
            config.freeze()
        random.seed(None)

        os.makedirs(config.OUTPUT, exist_ok=True)

        filename = "sel_num.data"
        if dist.get_rank() == 0:
            if not os.path.exists(sel_path := os.path.join(config.OUTPUT, filename)):
                sel_num = list(range(0, config.SET.COUNT))
                random.shuffle(sel_num)
                with open(os.path.join(config.OUTPUT, filename), "wb") as f:
                    pickle.dump(sel_num, f)
                del sel_num
        dist.barrier()

        with open(os.path.join(config.OUTPUT, filename), "rb") as f:
            sel_num = pickle.load(f)

        config.defrost()
        config.SET.TRAIN_INDEX = sel_num[0: int(round(0.8 * len(sel_num)))]
        config.SET.TEST_INDEX = sel_num[int(round(0.8 * len(sel_num))): len(sel_num)]
        config.freeze()

        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

        create_logger(
            logger,
            output_dir=config.OUTPUT,
            dist_rank=dist.get_rank(),
            name=f"{config.MODEL.NAME}",
        )

        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))

        main(config)
        logger.handlers.clear()
