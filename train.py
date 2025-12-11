# This code is constructed based on Pytorch Implementation of FixMatch(https://github.com/kekmodel/FixMatch-pytorch)
import argparse
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from utils import Logger
from progress.bar import Bar
from typing import Optional


logger = logging.getLogger(__name__)
best_acc = 0
best_acc_b = 0

def top2_gap(prob: torch.Tensor) -> torch.Tensor:
    v = prob.topk(2, dim=1).values
    return v[:, 0] - v[:, 1]

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * ( (p * (torch.log(p + eps) - torch.log(m + eps))).sum(1) +
                   (q * (torch.log(q + eps) - torch.log(m + eps))).sum(1) )

def scarce_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mean over valid positions; avoid divide-by-zero
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom



def make_imb_data(max_num, class_num, gamma, flag = 1, flag_LT = 0):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    return list(class_num_list)


def compute_adjustment_list(label_list, tro, args):
    label_freq_array = np.array(label_list)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments


def compute_py(train_loader, args):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell) in enumerate(train_loader):
        labell = labell.to(args.device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(args.device)
    return label_freq_array


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', epoch_p=1):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def compute_adjustment(train_loader, tro, args):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell) in enumerate(train_loader):
        labell = labell.to(args.device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments


def compute_adjustment_by_py(py, tro, args):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(args.device)
    return adjustments


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'smallimagenet'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext', 'resnet'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=250000, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=500, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    parser.add_argument('--num-max', default=500, type=int,
                        help='the max number of the labelled data')
    parser.add_argument('--num-max-u', default=4000, type=int,
                        help='the max number of the unlabeled data')
    parser.add_argument('--imb-ratio-label', default=1, type=int,
                        help='the imbalanced ratio of the labelled data')
    parser.add_argument('--imb-ratio-unlabel', default=1, type=int,
                        help='the imbalanced ratio of the unlabeled data')
    parser.add_argument('--flag-reverse-LT', default=0, type=int,
                        help='whether to reverse the distribution of the unlabeled data')
    parser.add_argument('--ema-mu', default=0.99, type=float,
                        help='mu when ema')

    parser.add_argument('--tau', default=2.0, type=float,
                        help='tau for (fixed) logit adjustment in both branches')
    parser.add_argument('--est-epoch', default=5, type=int,
                        help='the start step to estimate the distribution')
    parser.add_argument('--img-size', default=32, type=int,
                        help='image size for small imagenet')

    # === ours: modules & hyper-params ===
    parser.add_argument('--dwmm', action='store_true', help='enable DWMM')
    parser.add_argument('--dwmm_tau0', type=float, default=0.20, help='target margin for DWMM hinge/softplus')
    parser.add_argument('--dwmm_alpha', type=float, default=1.0, help='weighting alpha for JS in DWMM')
    parser.add_argument('--dwmm_beta', type=float, default=1.0, help='weighting beta for gap in DWMM')
    parser.add_argument('--dwmm_lambda', type=float, default=0.7, help='loss weight for DWMM')

    parser.add_argument('--ddvmix', action='store_true', help='enable DD-V-Mix')
    parser.add_argument('--ddvmix_lambda', type=float, default=0.5, help='loss weight for DD-V-Mix')
    parser.add_argument('--ddvmix_kappa', type=float, default=1.0, help='Beta(kappa,kappa) for vicinal lambda')

    parser.add_argument('--selcons', action='store_true', help='enable selective consistency (consensus-only KL)')
    parser.add_argument('--selcons_lambda_c', type=float, default=0.1, help='KL weight on consensus region')

    parser.add_argument('--tau_hi', type=float, default=0.85, help='high-confidence threshold')
    parser.add_argument('--tau_lo', type=float, default=0.60, help='low-confidence threshold')
    parser.add_argument('--gap_delta', type=float, default=0.15, help='boundary top1-top2 gap threshold')
    parser.add_argument('--proto_m', type=float, default=0.99, help='EMA momentum for class prototypes')


    args = parser.parse_args()
    global best_acc
    global best_acc_b

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'resnet':
            import models.resnet_ori as models
            model = models.ResNet50(num_classes=args.num_classes, rotation=True, classifier_bias=True)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.dataset_name = 'cifar-10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.dataset_name = 'cifar-100'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == 'stl10':
        args.num_classes = 10
        args.dataset_name = 'stl-10'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'svhn':
        args.num_classes = 10
        args.dataset_name = 'svhn'
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'smallimagenet':
        args.num_classes = 127
        if args.img_size == 32:
            args.dataset_name = 'imagenet32'
        elif args.img_size == 64:
            args.dataset_name = 'imagenet64'

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, 'datasets/'+args.dataset_name)

    if args.local_rank == 0:
        torch.distributed.barrier()

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    args.est_step = 0

    args.py_con = compute_py(labeled_trainloader, args)
    args.py_uni = torch.ones(args.num_classes) / args.num_classes
    args.py_rev = torch.flip(args.py_con, dims=[0])

    args.py_uni = args.py_uni.to(args.device)

    args.adjustment = compute_adjustment_by_py(args.py_con, args.tau, args)

    class_list = []
    for i in range(args.num_classes):
        class_list.append(str(i))

    title = 'FixMatch-' + args.dataset
    args.logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
    args.logger.set_names(['Top1 acc', 'Top5 acc', 'Best Top1 acc', 'Top1_b acc', 'Top5_b acc', 'Best Top1_b acc'])

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    args.u_py = torch.ones(args.num_classes) / args.num_classes
    args.u_py = args.u_py.to(args.device)

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.u_py = checkpoint['u_py']
        args.u_py = args.u_py.to(args.device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)
    args.logger.close()


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    global best_acc_b
    test_accs = []
    avg_time = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        print('current epoch: ', epoch+1)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        # 仅主进程显示进度条
        bar = Bar('Training', max=args.eval_step) if args.local_rank in [-1, 0] else None

        if epoch > args.est_epoch:
            print()
            # args.adjustment_l1 = compute_adjustment_by_py(args.py_con, tau, args)

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, inputs_u_s1), u_real = next(unlabeled_iter)

            u_real = u_real.cuda()
            mask_l = (u_real != -2)
            mask_l = mask_l.cuda()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1)), 3*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)

            logits_feat = model(inputs)
            logits = model.classify(logits_feat)

            logits = de_interleave(logits, 3*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s, logits_u_s1 = logits[batch_size:].chunk(3)
            del logits
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            logits_b = model.classify1(logits_feat)

            logits_b = de_interleave(logits_b, 3 * args.mu + 1)
            logits_x_b = logits_b[:batch_size]
            logits_u_w_b, logits_u_s_b, logits_u_s1_b = logits_b[batch_size:].chunk(3)
            del logits_b

            # === NEW: also de_interleave the features to pick unlabeled features ===
            feats = de_interleave(logits_feat, 3*args.mu+1)
            feats_x = feats[:batch_size]
            feats_u_w, feats_u_s, feats_u_s1 = feats[batch_size:].chunk(3)
            # 初始化/检查 prototypes
            if not hasattr(args, 'prototypes') or (args.prototypes is None):
                args.prototypes = torch.zeros(args.num_classes, feats_x.size(1), device=args.device)
                args.proto_cnt = torch.zeros(args.num_classes, device=args.device)

            Lx_b = F.cross_entropy(logits_x_b + args.adjustment, targets_x, reduction='mean')

            pseudo_label = torch.softmax((logits_u_w.detach() - args.adjustment) / args.T, dim=-1)
            pseudo_label_b = torch.softmax(logits_u_w_b.detach() / args.T, dim=-1)
            pseudo_label_t = torch.softmax(logits_u_w.detach() / args.T, dim=-1)

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            max_probs_b, targets_u_b = torch.max(pseudo_label_b, dim=-1)
            max_probs_t, targets_u_t = torch.max(pseudo_label_t, dim=-1)

            mask = max_probs.ge(args.threshold)
            mask_b = max_probs_b.ge(args.threshold)
            mask_t = max_probs_t.ge(args.threshold)

            mask_ss_b_h2 = mask_b + mask  # 替换原来的：mask_ss_b_h2 = mask_b + mask_h2
            mask_ss_t = mask + mask_t  # 保持不变

            mask = mask.float()
            mask_b = mask_b.float()

            mask_ss_b_h2 = mask_ss_b_h2.float()
            mask_ss_t = mask_ss_t.float()

            mask_twice_ss_b_h2 = torch.cat([mask_ss_b_h2, mask_ss_b_h2], dim=0).cuda()
            mask_twice_ss_t = torch.cat([mask_ss_t, mask_ss_t], dim=0).cuda()

            logits_u_s_twice = torch.cat([logits_u_s, logits_u_s1], dim=0).cuda()
            targets_u_twice = torch.cat([targets_u, targets_u], dim=0).cuda()
            logits_u_s_b_twice = torch.cat([logits_u_s_b, logits_u_s1_b], dim=0).cuda()

            now_mask = torch.zeros(args.num_classes)
            now_mask = now_mask.to(args.device)
            u_real[u_real==-2] = 0

            # Lu = (F.cross_entropy(logits_u_s_twice, targets_u_twice,
            #                       reduction='none') * mask_twice_ss_t).mean()
            # Lu_b = (F.cross_entropy(logits_u_s_b_twice, targets_u_twice,
            #                         reduction='none') * mask_twice_ss_b_h2).mean()
            #
            # loss = Lx + Lu + Lx_b + Lu_b
            # -------------------- OUR MODULES START --------------------
            # ===== 统一概率、置信度、边界性判定 =====
            pA_w = torch.softmax((logits_u_w.detach() - args.adjustment) / args.T, dim=1)   # head A (your 'standard' head)
            pB_w = torch.softmax(logits_u_w_b.detach() / args.T, dim=1)                     # head B (your 'balanced' head)
            sA, cA = pA_w.max(1); sB, cB = pB_w.max(1)
            gapA = top2_gap(pA_w); gapB = top2_gap(pB_w)
            gap_min = torch.minimum(gapA, gapB)

            tau_hi, tau_lo, delta = args.tau_hi, args.tau_lo, args.gap_delta
            lowlow = (torch.maximum(sA, sB) < tau_lo)
            leanA  = (sA >= tau_hi) & (sB < tau_hi)
            leanB  = (sB >= tau_hi) & (sA < tau_hi)
            # 强边界：双高不同类 + 小间距
            bdboth = (sA >= tau_hi) & (sB >= tau_hi) & (cA != cB) & (gap_min <= delta)
            # 弱边界：一高一低 + 小间距 + 另一头 top2 包含高置信头的 top1
            cross_AinB = (pB_w.topk(2,1).indices == cA[:,None]).any(1)
            cross_BinA = (pA_w.topk(2,1).indices == cB[:,None]).any(1)
            weak_bdry = ((leanA & cross_AinB) | (leanB & cross_BinA)) & (gap_min <= delta)

            # ====== 原有 supervised/unlabeled CE (保持原样) ======
            Lu = (F.cross_entropy(logits_u_s_twice, targets_u_twice, reduction='none') * mask_twice_ss_t).mean()
            Lu_b = (F.cross_entropy(logits_u_s_b_twice, targets_u_twice, reduction='none') * mask_twice_ss_b_h2).mean()

            # ====== DWMM: 分歧加权的最大间隔 ======
            L_dwmm = torch.tensor(0., device=args.device)
            if args.dwmm:
                # JS & gap 归一作为权重
                JS = js_divergence(pA_w, pB_w)
                JSn  = (JS - JS.mean()) / (JS.std() + 1e-6)
                gapn = (gap_min - gap_min.mean()) / (gap_min.std() + 1e-6)
                w_dw = torch.sigmoid(args.dwmm_alpha * JSn - args.dwmm_beta * gapn).detach()

                # 使用融合 logit 计算 logit-margin
                z_co_w = 0.5 * (logits_u_w + logits_u_w_b)  # 简洁起步；可换成 α(x) 动态融合
                top2 = z_co_w.topk(2, dim=1)
                margin = top2.values[:,0] - top2.values[:,1]
                mm_hinge = torch.clamp(args.dwmm_tau0 - margin, min=0.0)
                mask_dw = (bdboth | weak_bdry).float()
                # 稀缺归一
                L_dwmm = scarce_mean(w_dw * mm_hinge, mask_dw)

            # ====== DD-V-Mix: 沿边界 vicinal / mixup（特征空间） ======
            L_mix = torch.tensor(0., device=args.device)
            if args.ddvmix:
                idx = (bdboth | weak_bdry).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    # 取特征 & 类原型；首次已初始化 args.prototypes
                    muA = args.prototypes[cA[idx]]
                    muB = args.prototypes[cB[idx]]
                    # Beta(kappa,kappa) or靠近0.5（随JS/gap可调，这里用常数/无梯度）
                    lam = torch.full((idx.numel(),), 0.5, device=args.device)
                    # 邻域插值：当前样本特征 vs 两类原型均值
                    h_cur = feats_u_w[idx]
                    h_mix = lam[:,None]*h_cur + (1-lam[:,None]) * 0.5*(muA + muB)
                    # 经过分类头（任一头都可，这里用 head A）
                    z_mix = model.classify(h_mix)
                    logp_mix = F.log_softmax(z_mix, dim=1)
                    # 双类软目标
                    q = torch.zeros_like(z_mix)
                    q.scatter_(1, cA[idx,None], lam[:,None])
                    q.scatter_(1, cB[idx,None], (1-lam)[:,None])
                    L_mix = F.kl_div(logp_mix, q, reduction='batchmean')

            # ====== Selective Consistency：只在共识高置信区对齐 ======
            L_selc = torch.tensor(0., device=args.device)
            if args.selcons:
                consensus = (cA == cB) & (torch.minimum(sA, sB) >= tau_hi)
                if consensus.any():
                    pa = F.log_softmax(logits_u_w[consensus], dim=1)
                    pb = F.softmax(logits_u_w_b[consensus].detach(), dim=1)
                    L_ab = F.kl_div(pa, pb, reduction='batchmean')
                    pb2 = F.log_softmax(logits_u_w_b[consensus], dim=1)
                    pa2 = F.softmax(logits_u_w[consensus].detach(), dim=1)
                    L_ba = F.kl_div(pb2, pa2, reduction='batchmean')
                    L_selc = 0.5 * (L_ab + L_ba) * args.selcons_lambda_c

            # ====== 原型 EMA 更新（用标注 + 可靠伪标签的样本） ======
            with torch.no_grad():
                m = args.proto_m
                # labeled
                for cls in targets_x.unique():
                    cls = int(cls.item())
                    mask_c = (targets_x == cls)
                    if mask_c.any():
                        mean_c = feats_x[mask_c].mean(0)
                        args.prototypes[cls] = m*args.prototypes[cls] + (1-m)*mean_c
                # 可靠伪标签（共识或可靠 lean）
                reliable_leanA = leanA & cross_AinB & (gap_min <= delta)
                reliable_leanB = leanB & cross_BinA & (gap_min <= delta)
                reliable_u = ( (cA==cB) & (torch.minimum(sA,sB)>=tau_hi) ) | reliable_leanA | reliable_leanB
                if reliable_u.any():
                    cls_idx = torch.where(reliable_u, cA, cB)  # 共识用 cA==cB，其余取高置信那头的类
                    for cls in cls_idx.unique():
                        cls = int(cls.item())
                        mask_c = reliable_u & (cls_idx==cls)
                        if mask_c.any():
                            mean_c = feats_u_w[mask_c].mean(0)
                            args.prototypes[cls] = m*args.prototypes[cls] + (1-m)*mean_c

            # ====== 汇总无标块 & 总损失 ======
            loss_u_block = Lu + Lu_b + args.dwmm_lambda * L_dwmm + args.ddvmix_lambda * L_mix + L_selc
            loss = Lx + Lx_b + loss_u_block
            # -------------------- OUR MODULES END --------------------

            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item()+Lx_b.item())
            losses_u.update(Lu.item()+Lu_b.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())

            # 统计耗时
            batch_time.update(time.time() - end)
            end = time.time()

            # 更新进度条内容（仅主进程）
            if bar is not None:
                bar.suffix = (
                    '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total} | ETA: {eta} | '
                    'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | '
                    'Loss_u: {loss_u:.4f}'
                ).format(
                    batch=batch_idx + 1,
                    size=args.eval_step,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                )
                bar.next()

        if bar is not None:
            bar.finish()
        print('\n')

        avg_time.append(batch_time.avg)

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_top5_acc, test_acc_b, test_top5_acc_b = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_b, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc_b > best_acc_b

            best_acc = max(test_acc, best_acc)
            best_acc_b = max(test_acc_b, best_acc_b)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            if (epoch+1) % 10 == 0 or (is_best and epoch > 250):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': test_acc,
                    'best_acc': best_acc_b,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'u_py': args.u_py,
                }, is_best, args.out, epoch_p=epoch+1)

            test_accs.append(test_acc_b)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc_b))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            args.logger.append([test_acc, test_top5_acc, best_acc, test_acc_b, test_top5_acc_b, best_acc_b])
    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_b = AverageMeter()
    top5_b = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs_feat = model(inputs)
            outputs = model.classify(outputs_feat)
            outputs_b = model.classify1(outputs_feat)
            loss = F.cross_entropy(outputs_b, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            prec1_b, prec5_b = accuracy(outputs_b, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            top1_b.update(prec1_b.item(), inputs.shape[0])
            top5_b.update(prec5_b.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info("top-1 acc: {:.2f}".format(top1_b.avg))
    logger.info("top-5 acc: {:.2f}".format(top5_b.avg))

    return losses.avg, top1.avg, top5.avg, top1_b.avg, top5_b.avg


if __name__ == '__main__':
    main()
