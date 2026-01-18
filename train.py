# This code is constructed based on Pytorch Implementation of FixMatch(https://github.com/kekmodel/FixMatch-pytorch)
import argparse
import logging
import math
import os
import random
import shutil
import time
import csv
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

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_b = 0


def js_divergence(p, q, eps=1e-12):
    # p,q: [B,C] prob
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)  # [B]


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

    parser.add_argument('--tau1', default=2, type=float,
                        help='tau for head1 consistency')
    parser.add_argument('--tau12', default=2, type=float,
                        help='tau for head2 consistency')
    parser.add_argument('--tau2', default=2, type=float,
                        help='tau for head2 balanced CE loss')
    parser.add_argument('--ema-u', default=0.9, type=float,
                        help='ema ratio for estimating distribution of the unlabeled data')
    parser.add_argument('--est-epoch', default=40, type=int,
                        help='the start step to estimate the distribution')
    parser.add_argument('--dwsc-lambda', default=0.10, type=float,
                        help='lambda for disagreement-weighted soft consistency (DW-SC), starts at est-epoch')
    parser.add_argument('--dwsc-warm-epochs', default=80, type=int,
                        help='warmup epochs for DW-SC lambda ramp (starts at est-epoch)')
    parser.add_argument('--dwct-class-alpha', default=0.05, type=float,
                        help='class-aware threshold scaling for pseudo-label masks (0 disables)')
    parser.add_argument('--dwct-js-self-th', default=0.05, type=float,
                        help='JS threshold for enabling head2 self-training when heads agree')
    parser.add_argument('--dwct-self-lambda', default=0.25, type=float,
                        help='weight for head2 self-training loss (only on safe-agree samples)')

    parser.add_argument('--img-size', default=32, type=int,
                        help='image size for small imagenet')

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
        # ---- disagreement diagnostics logging (epoch-level CSV) ----
        args.diag_csv = os.path.join(args.out, "disagreement_diagnostics.csv")
        if not os.path.exists(args.diag_csv):
            with open(args.diag_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch","tau_curr",
                    "mask1_rate","mask2_rate","both_conf_rate","disagree_rate","conflict_both_rate",
                    "js_w_mean","js_w_q50","js_w_q90","js_w_q99",
                    "js_s_mean","agree_h1_ws","agree_h2_ws",
                    "pl_acc_h1_on_u_real","pl_acc_h2_on_u_real",
                    "pl_acc_h1_conf_on_u_real","pl_acc_h2_conf_on_u_real",
                    "conflict_any_rate",
                    "pl_acc_conflict_any_on_u_real","pl_acc_conflict_both_on_u_real",
                    "pl_acc_conflict_any_conf_on_u_real","pl_acc_conflict_both_conf_on_u_real",
                    "L_conflict","L_soft","Lu_h1","Lu_h2","Lu_total",
                ])


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

    args.adjustment_l1 = compute_adjustment_by_py(args.py_con, 1.0, args)
    args.adjustment_l12 = compute_adjustment_by_py(args.py_con, 1.0, args)
    args.adjustment_l2 = compute_adjustment_by_py(args.py_con, args.tau2, args)

    args.taumin = 0
    args.taumax = args.tau1

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

    args.u_py = args.py_con.clone().detach().float().to(args.device)
    args.u_py = args.u_py / (args.u_py.sum() + 1e-12)
    args.tau_curr = 0.0
    args.acralign_epoch = False

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

    # count_KL is used to accumulate symmetric-KL distances for adaptive tau
    count_KL = torch.zeros(3).to(args.device)

    KL_div = nn.KLDivLoss(reduction='sum')

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        print('current epoch: ', epoch+1)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        # ---- disagreement diagnostics (reset per epoch) ----
        js_w_meter = AverageMeter()
        js_s_meter = AverageMeter()
        disagree_meter = AverageMeter()
        conflict_both_meter = AverageMeter()
        mask1_meter = AverageMeter()
        mask2_meter = AverageMeter()
        both_conf_meter = AverageMeter()
        agree_h1_ws_meter = AverageMeter()
        agree_h2_ws_meter = AverageMeter()
        pl_acc_h1_meter = AverageMeter()
        pl_acc_h2_meter = AverageMeter()
        pl_acc_h1_conf_meter = AverageMeter()
        pl_acc_h2_conf_meter = AverageMeter()

        conflict_any_meter = AverageMeter()
        pl_acc_conflict_any_meter = AverageMeter()
        pl_acc_conflict_both_meter = AverageMeter()
        pl_acc_conflict_any_conf_meter = AverageMeter()
        pl_acc_conflict_both_conf_meter = AverageMeter()
        L_conflict_meter = AverageMeter()
        L_soft_meter = AverageMeter()
        Lu_h1_meter = AverageMeter()
        Lu_h2_meter = AverageMeter()
        Lu_total_meter = AverageMeter()
        # buffers for quantiles (store CPU tensors, small overhead)
        js_w_buf = []


        # 仅主进程显示进度条
        bar = Bar('Training', max=args.eval_step) if args.local_rank in [-1, 0] else None

        if epoch > args.est_epoch:
            # ---- warm-start for tau update ----
            # In the first epoch after est_epoch, KL statistics (count_KL) may still be all zeros
            # because they are only accumulated when epoch > est_epoch. If we apply the tau update
            # with count_KL=0, tau becomes taumax/3 (e.g., 2/3) and can introduce an unnecessary
            # training-mode perturbation. We instead initialize count_KL from the current u_py estimate.
            if float(count_KL.abs().sum().item()) < 1e-12:
                with torch.no_grad():
                    u_py = args.u_py.detach()
                    u_py = u_py / (u_py.sum() + 1e-12)
                    py_con = args.py_con.detach()
                    py_uni = args.py_uni.detach()
                    py_rev = args.py_rev.detach()
                    KL_con = 0.5 * KL_div(py_con.log(), u_py) + 0.5 * KL_div(u_py.log(), py_con)
                    KL_uni = 0.5 * KL_div(py_uni.log(), u_py) + 0.5 * KL_div(u_py.log(), py_uni)
                    KL_rev = 0.5 * KL_div(py_rev.log(), u_py) + 0.5 * KL_div(u_py.log(), py_rev)
                    count_KL = torch.stack([KL_con, KL_uni, KL_rev]) * float(args.eval_step)

            count_KL = count_KL / args.eval_step
            KL_softmax = (torch.exp(count_KL[0])) / (torch.exp(count_KL[0])+torch.exp(count_KL[1])+torch.exp(count_KL[2]))
            tau = args.taumin + (args.taumax - args.taumin) * KL_softmax
            if math.isnan(tau)==False:
                args.tau_curr = float(tau)
                args.adjustment_l1 = compute_adjustment_by_py(args.py_con, tau, args)
                # Keep teacher debias strength (tau12) consistent with current tau trajectory, while respecting args.tau12 as an upper bound.
                tau12_curr = min(float(getattr(args, 'tau12', 1.0)), 1.0 + 0.5 * float(tau))
                args.adjustment_l12 = compute_adjustment_by_py(args.py_con, tau12_curr, args)
                # Decide ACR/CCL-style mode *dynamically* based on current tau trajectory.
                # This prevents 'tau=222' from forcing aggressive behavior in consistent setting.
                user_acr = (float(getattr(args, 'tau1', 1.0)) > 1.0) or (float(getattr(args, 'tau12', 1.0)) > 1.0)
                args.acralign_epoch = bool(user_acr and (float(getattr(args, 'tau_curr', 0.0)) >= 0.5))

        count_KL = torch.zeros(3).to(args.device)

        if epoch <= args.est_epoch:
            args.acralign_epoch = False

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

            u_real = u_real.to(args.device)
            mask_l = (u_real != -2)
            mask_l = mask_l.to(args.device)

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
            Lx_b = F.cross_entropy(logits_x_b + args.adjustment_l2, targets_x, reduction='mean')

            # ---- posteriors (debiased/raw) ----
            pseudo_label = torch.softmax((logits_u_w.detach() - args.adjustment_l1) / args.T, dim=-1)   # head1 debiased (for head1 PL)
            pseudo_label_t = torch.softmax(logits_u_w.detach() / args.T, dim=-1)                        # head1 raw (for disagreement)
            pseudo_label_h2_std = torch.softmax((logits_u_w.detach() - args.adjustment_l12) / args.T, dim=-1)  # head1-alt teacher (for head2)
            # head2 raw posterior (for disagreement)
            pseudo_label_b_raw = torch.softmax((logits_u_w_b.detach()) / args.T, dim=-1)

            # head2 posterior on weak unlabeled (for masks / training):
            # - AF53-mode (tau1=tau12=1): use debiased posterior (+adjustment_l2) -> best consistent
            # - ACR/CCL-mode (tau1 or tau12 > 1): use raw posterior for better calibration on uniform/reverse
            acralign = bool(getattr(args, 'acralign_epoch', False))
            if acralign:
                pseudo_label_b = pseudo_label_b_raw
            else:
                pseudo_label_b = torch.softmax((logits_u_w_b.detach() + args.adjustment_l2) / args.T, dim=-1)

            # ---- disagreement is computed on RAW posteriors to decouple from logit-adjustment ----
            with torch.no_grad():
                # js_w = js_divergence(pseudo_label_t, pseudo_label_b_raw) / (math.log(args.num_classes) + 1e-12)
                js_w = js_divergence(pseudo_label_t, pseudo_label_b_raw) / (math.log(args.num_classes) + 1e-12)
                js_n = js_w.clamp(0.0, 1.0)

            # ---- JS-weighted soft teacher for head2 (committee fusion) ----
            # JS small -> trust head1 teacher; JS large -> lean more on head2 posterior.
            w_mix = js_n.detach()
            pseudo_label_h2 = (1.0 - w_mix).unsqueeze(1) * pseudo_label_h2_std + w_mix.unsqueeze(1) * pseudo_label_b


            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            max_probs_h2, targets_u_h2 = torch.max(pseudo_label_h2, dim=-1)
            max_probs_b, targets_u_b = torch.max(pseudo_label_b, dim=-1)
            max_probs_b_raw, targets_u_b_raw = torch.max(pseudo_label_b_raw, dim=-1)
            max_probs_t, targets_u_t = torch.max(pseudo_label_t, dim=-1)

            # 1) binary masks (bool)
            # ===================== Disagreement-Weighted Cross-Teaching (DWCT) =====================
            # Key idea:
            #   - Disagreement is a *risk* signal in SSL (high epistemic uncertainty) -> downweight hard pseudo-label CE.
            #   - Keep teacher/mask aligned: head2 learns from head1-alt teacher when using maskh2.
            #   - No conflict-specific loss (avoid pulling heads to an incorrect "average" early).
            #
            # Hyper-params kept minimal:
            #   - warmup: reuse args.est_epoch (no new CLI)
            #   - gamma fixed to 2 (stronger downweight on high disagreement)
            #   - head2 self-training disabled by default (set self_lambda>0 if you want)

            # 1) confidence masks
            # Optional: class-aware thresholds to increase tail pseudo-label coverage (default off).
            # Set args.dwct_class_alpha in [0.1, 0.5] to enable (tail gets lower tau).
            alpha_tau = float(getattr(args, 'dwct_class_alpha', 0.0))
            if alpha_tau > 0 and hasattr(args, 'py_con') and args.py_con is not None:
                with torch.no_grad():
                    py = args.py_con.to(max_probs.device).float()
                    tmin, tmax = py.min(), py.max()
                    tailness_table = 1.0 - (py - tmin) / (tmax - tmin + 1e-12)  # head:0, tail:1
                    tau_base = float(args.threshold)
                    tau_min = float(getattr(args, 'dwct_tau_min', 0.70))
                    tau_max = float(getattr(args, 'dwct_tau_max', 0.98))
                    tau1  = (tau_base * (1.0 - alpha_tau * tailness_table[targets_u])).clamp(tau_min, tau_max)
                    tau2  = (tau_base * (1.0 - alpha_tau * tailness_table[targets_u_b])).clamp(tau_min, tau_max)
                    taut  = (tau_base * (1.0 - alpha_tau * tailness_table[targets_u_t])).clamp(tau_min, tau_max)
                    tauh2 = (tau_base * (1.0 - alpha_tau * tailness_table[targets_u_h2])).clamp(tau_min, tau_max)
                mask1 = max_probs.ge(tau1)          # head1 (debiased) confident
                mask2 = max_probs_b.ge(tau2)        # head2 confident
                maskt = max_probs_t.ge(taut)        # head1 (raw) confident
                maskh2 = max_probs_h2.ge(tauh2)     # head1-alt teacher confident
            else:
                mask1 = max_probs.ge(args.threshold)          # head1 (debiased) confident
                mask2 = max_probs_b.ge(args.threshold)        # head2 confident
                maskt = max_probs_t.ge(args.threshold)        # head1 (raw) confident
                maskh2 = max_probs_h2.ge(args.threshold)      # head1-alt teacher confident

            # Supervision weight mode:
            # - AF53-mode: boolean OR -> weights in {0,1} (best consistent)
            # - ACR/CCL-mode: addition -> weights in {0,1,2} (better coverage on uniform/reverse)
            acralign = bool(getattr(args, 'acralign_epoch', False))
            if acralign:
                mask_sup_h1 = (mask1.float() + maskt.float())        # head1 hard-PL weight in {0,1,2}
                mask_sup_h2_ct = (mask2.float() + maskh2.float())    # head2 cross-teaching weight in {0,1,2}
                mask_sup_h2_self = mask2.float()
            else:
                mask_sup_h1 = (mask1 | maskt)                        # head1 hard-PL eligibility
                mask_sup_h2_ct = maskh2                              # head2 cross-teaching eligibility
                mask_sup_h2_self = mask2                             # head2 self-training eligibility

            # 2) disagreement (normalized JS in [0,1]) and reliability weight
            with torch.no_grad():
                # js_w/js_n were computed above on RAW posteriors to decouple disagreement from logit-adjustment.
                conf_mean = 0.5 * (max_probs.detach() + max_probs_b.detach())
                w_soft = (js_n * (1.0 - js_n) * conf_mean).clamp(0.0, 1.0)
                # We keep hard-PL CE weights unchanged by disagreement (baseline behavior).
                w_dis_eff = torch.ones_like(js_n)
            # ---- disagreement diagnostics (per step, epoch-averaged) ----
            with torch.no_grad():
                # Disagreement on RAW predictions (decoupled from logit-adjustment)
                disagree = (targets_u_t != targets_u_b_raw)
                both_conf = (mask1 & mask2)
                conflict_both = both_conf & disagree
                conflict_any = disagree & (mask1 | mask2)

                # rates
                mask1_meter.update(mask1.float().mean().item())
                mask2_meter.update(mask2.float().mean().item())
                both_conf_meter.update(both_conf.float().mean().item())
                disagree_meter.update(disagree.float().mean().item())
                conflict_both_meter.update(conflict_both.float().mean().item())
                conflict_any_meter.update(conflict_any.float().mean().item())
                js_w_meter.update(js_n.mean().item())
                js_w_buf.append(js_n.detach().cpu())

                # weak->strong agreement
                p1_s = torch.softmax((logits_u_s.detach() - args.adjustment_l1) / args.T, dim=-1)
                p2_s = torch.softmax((logits_u_s_b.detach() + args.adjustment_l2) / args.T, dim=-1)
                pred1_s = p1_s.argmax(dim=-1)
                pred2_s = p2_s.argmax(dim=-1)
                agree_h1_ws_meter.update((pred1_s == targets_u).float().mean().item())
                agree_h2_ws_meter.update((pred2_s == targets_u_b).float().mean().item())
                js_s = js_n.detach()
                js_s_meter.update(js_s.mean().item())

                # pseudo-label accuracy on u_real subset (if available)
                if mask_l.any():
                    acc1 = (targets_u == u_real).float()
                    acc2 = (targets_u_b == u_real).float()
                    pl_acc_h1_meter.update(acc1[mask_l].mean().item())
                    pl_acc_h2_meter.update(acc2[mask_l].mean().item())
                    if (mask_l & mask1).any():
                        pl_acc_h1_conf_meter.update(acc1[mask_l & mask1].mean().item())
                    if (mask_l & mask2).any():
                        pl_acc_h2_conf_meter.update(acc2[mask_l & mask2].mean().item())
                    # conflict buckets
                    if (mask_l & conflict_any).any():
                        pl_acc_conflict_any_meter.update(acc1[mask_l & conflict_any].mean().item())
                    if (mask_l & conflict_both).any():
                        pl_acc_conflict_both_meter.update(acc1[mask_l & conflict_both].mean().item())
                    if (mask_l & conflict_any & mask1).any():
                        pl_acc_conflict_any_conf_meter.update(acc1[mask_l & conflict_any & mask1].mean().item())
                    if (mask_l & conflict_both & mask1).any():
                        pl_acc_conflict_both_conf_meter.update(acc1[mask_l & conflict_both & mask1].mean().item())



            # ---- dynamic tau bookkeeping (AF53 <-> ACR/CCL merge) ----
            # We update an EMA estimate of unlabeled label distribution u_py, then compute tau via symmetric-KL distances.
            # Key difference between AF53 and 86e0:
            #   - AF53 (best consistent): estimate u_py ONLY from safe+agree pseudo-labels (very conservative).
            #   - 86e0 (best uniform/reverse): estimate u_py from representative head2 confident samples, downweighting high-JS.
            # We merge them WITHOUT adding new CLI switches:
            #   - If tau1<=1 and tau12<=1: run AF53 estimation (preserves your best consistent result).
            #   - Otherwise: run 86e0 estimation (improves uniform/reverse).
            if epoch > args.est_epoch:
                with torch.no_grad():
                    args.est_step = getattr(args, "est_step", 0) + 1
                    acralign = bool(getattr(args, 'acralign_epoch', False))

                    if acralign:
                        # Representative estimation: head2 confident samples, reliability decreases with disagreement (JS)
                        est_rel = (1.0 - 0.5 * js_n.detach()).clamp(0.2, 1.0).float()
                        est_w = mask2.float() * est_rel
                        est_targets = targets_u_b_raw.long()
                    else:
                        # Conservative estimation: only when both heads confident and agree
                        # Match original AF53 behavior: estimate from u_real-available subset to stabilize tau in consistent
                        est_w = (mask_l & mask1 & mask2 & (targets_u_t == targets_u_b_raw)).float()
                        est_targets = targets_u_b_raw.long()

                    if est_w.sum() > 0:
                        now_mask = torch.zeros(args.num_classes, device=args.device, dtype=est_w.dtype)
                        now_mask.index_add_(0, est_targets, est_w.to(now_mask.dtype))
                        if now_mask.sum() > 0:
                            now_mask = now_mask / (now_mask.sum() + 1e-12)
                            args.u_py = args.ema_u * args.u_py + (1 - args.ema_u) * now_mask

                            # symmetric KL to three candidate priors
                            u_py = args.u_py.clamp(min=1e-12)
                            py_con = args.py_con.clamp(min=1e-12)
                            py_uni = args.py_uni.clamp(min=1e-12)
                            py_rev = args.py_rev.clamp(min=1e-12)
                            KL_con = 0.5 * KL_div(py_con.log(), u_py) + 0.5 * KL_div(u_py.log(), py_con)
                            KL_uni = 0.5 * KL_div(py_uni.log(), u_py) + 0.5 * KL_div(u_py.log(), py_uni)
                            KL_rev = 0.5 * KL_div(py_rev.log(), u_py) + 0.5 * KL_div(u_py.log(), py_rev)
                            count_KL[0] = count_KL[0] + KL_con
                            count_KL[1] = count_KL[1] + KL_uni
                            count_KL[2] = count_KL[2] + KL_rev

            # head2 safe self-training gate (only after warmup)
            js_n = js_w.clamp(0.0, 1.0)
            js_self_th = float(getattr(args, 'dwct_js_self_th', 0.20))
            acralign = bool(getattr(args, 'acralign_epoch', False))

            # AF53-mode: safe head2 self-training (helps consistent).
            # ACR/CCL-mode: disable head2 self-confirmation by default (as in ACR/86e0) to improve uniform/reverse stability.
            if acralign:
                self_lambda = 0.0
                self_mask = mask2
            else:
                self_lambda = float(getattr(args, 'dwct_self_lambda', 0.25)) if epoch >= getattr(args, 'est_epoch', 0) else 0.0
                agree_top1 = (targets_u_t == targets_u_b_raw)
                self_mask = (mask2 & mask1 & agree_top1 & (js_n <= js_self_th))

            # weights (per-sample)
            w_h1      = mask_sup_h1.float()      * w_dis_eff
            w_h2_ct   = mask_sup_h2_ct.float()   * w_dis_eff
            w_h2_self = self_mask.float()        * w_dis_eff

            # twice weights/targets for two strong views
            w_h1_twice      = torch.cat([w_h1, w_h1], dim=0)
            w_h2_ct_twice   = torch.cat([w_h2_ct, w_h2_ct], dim=0)
            w_h2_self_twice = torch.cat([w_h2_self, w_h2_self], dim=0)
            targets_u_h2_twice = torch.cat([targets_u_h2, targets_u_h2], dim=0)
            targets_u_twice = torch.cat([targets_u, targets_u], dim=0)
            targets_u_b_twice = torch.cat([targets_u_b, targets_u_b], dim=0)

            logits_u_s_twice = torch.cat([logits_u_s, logits_u_s1], dim=0)
            logits_u_s_b_twice = torch.cat([logits_u_s_b, logits_u_s1_b], dim=0)


            # FixMatch-style hard CE losses
            if acralign:
                # 86e0-style: raw logits on strong views (robust in uniform/reverse when priors differ)
                Lu = (F.cross_entropy(logits_u_s_twice, targets_u_twice, reduction='none') * w_h1_twice).mean()
                Lu_b_ct = (F.cross_entropy(logits_u_s_b_twice, targets_u_h2_twice, reduction='none') * w_h2_ct_twice).mean()
                Lu_b = Lu_b_ct  # no head2 self-confirmation by default
            else:
                # AF53-style: debiased logits on strong views (best consistent in your experiments)
                Lu = (F.cross_entropy(logits_u_s_twice - args.adjustment_l1, targets_u_twice, reduction='none') * w_h1_twice).mean()
                Lu_b_ct = (F.cross_entropy(logits_u_s_b_twice + args.adjustment_l2, targets_u_h2_twice, reduction='none') * w_h2_ct_twice).mean()
                Lu_b_self = (F.cross_entropy(logits_u_s_b_twice + args.adjustment_l2, targets_u_b_twice, reduction='none') * w_h2_self_twice).mean()
                Lu_b = Lu_b_ct + self_lambda * Lu_b_self

            # DW-SC: disagreement-weighted soft consistency (committee posterior) on strong views
            dwsc_lambda = float(getattr(args, 'dwsc_lambda', 0.10))
            est_ep = int(getattr(args, 'est_epoch', 0))
            warm_ep = int(getattr(args, 'dwsc_warm_epochs', 10))
            if epoch < est_ep:
                dwsc_lambda_eff = 0.0
            elif warm_ep <= 0:
                dwsc_lambda_eff = dwsc_lambda
            else:
                prog = min(1.0, float(epoch - est_ep + 1) / float(warm_ep))
                dwsc_lambda_eff = dwsc_lambda * prog
            if dwsc_lambda_eff > 0:
                with torch.no_grad():
                    q_w = 0.5 * (pseudo_label + pseudo_label_b)
                    q_w = (q_w / (q_w.sum(dim=-1, keepdim=True) + 1e-12)).detach()
                    q_w_twice = torch.cat([q_w, q_w], dim=0)
                    w_soft_twice = torch.cat([w_soft, w_soft], dim=0)
                acralign = bool(getattr(args, 'acralign_epoch', False))
                if acralign:
                    logp1_s = F.log_softmax((logits_u_s_twice / args.T), dim=-1)
                    logp2_s = F.log_softmax((logits_u_s_b_twice / args.T), dim=-1)
                else:
                    logp1_s = F.log_softmax((logits_u_s_twice - args.adjustment_l1) / args.T, dim=-1)
                    logp2_s = F.log_softmax((logits_u_s_b_twice + args.adjustment_l2) / args.T, dim=-1)
                kl1 = F.kl_div(logp1_s, q_w_twice, reduction='none').sum(dim=-1)
                kl2 = F.kl_div(logp2_s, q_w_twice, reduction='none').sum(dim=-1)
                L_soft = ((kl1 + kl2) * 0.5 * w_soft_twice).mean()
            else:
                L_soft = torch.zeros((), device=logits_u_s_twice.device)

            # keep for logging compatibility
            L_conflict = torch.zeros((), device=logits_u_s_twice.device)
            Lu_total = Lu + Lu_b + dwsc_lambda_eff * L_soft

            # for logging only: effective hard-PL pass rate for head1
            sup_h1 = (w_h1 > 0).float()

            # ---- loss diagnostics ----
            with torch.no_grad():
                L_conflict_meter.update(float(L_conflict.detach().item()))
                L_soft_meter.update(float(L_soft.detach().item()))
                Lu_h1_meter.update(float(Lu.detach().item()))
                Lu_h2_meter.update(float(Lu_b.detach().item()))
                Lu_total_meter.update(float(Lu_total.detach().item()))

            # logging: mask = effective hard-PL rate (excluding conflict_both)
            mask_probs.update(sup_h1.mean().item())

            loss = Lx + Lx_b + Lu_total
            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item()+Lx_b.item())
            losses_u.update(Lu_total.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

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
            # ---- disagreement diagnostics: log to TensorBoard + CSV ----
            if len(js_w_buf) > 0:
                js_all = torch.cat(js_w_buf, dim=0)
                js_q50 = torch.quantile(js_all, 0.50).item()
                js_q90 = torch.quantile(js_all, 0.90).item()
                js_q99 = torch.quantile(js_all, 0.99).item()
            else:
                js_q50 = js_q90 = js_q99 = 0.0

            args.writer.add_scalar('diag/mask1_rate', mask1_meter.avg, epoch)
            args.writer.add_scalar('diag/mask2_rate', mask2_meter.avg, epoch)
            args.writer.add_scalar('diag/both_conf_rate', both_conf_meter.avg, epoch)
            args.writer.add_scalar('diag/disagree_rate', disagree_meter.avg, epoch)
            args.writer.add_scalar('diag/conflict_both_rate', conflict_both_meter.avg, epoch)
            args.writer.add_scalar('diag/js_w_mean', js_w_meter.avg, epoch)
            args.writer.add_scalar('diag/js_w_q90', js_q90, epoch)
            args.writer.add_scalar('diag/js_s_mean', js_s_meter.avg, epoch)
            args.writer.add_scalar('diag/agree_h1_ws', agree_h1_ws_meter.avg, epoch)
            args.writer.add_scalar('diag/agree_h2_ws', agree_h2_ws_meter.avg, epoch)
            args.writer.add_scalar('diag/pl_acc_h1_on_u_real', pl_acc_h1_meter.avg, epoch)
            args.writer.add_scalar('diag/pl_acc_h2_on_u_real', pl_acc_h2_meter.avg, epoch)
            args.writer.add_scalar('diag/pl_acc_h1_conf_on_u_real', pl_acc_h1_conf_meter.avg, epoch)
            args.writer.add_scalar('diag/pl_acc_h2_conf_on_u_real', pl_acc_h2_conf_meter.avg, epoch)

            args.writer.add_scalar('diag/conflict_any_rate', conflict_any_meter.avg, epoch)
            args.writer.add_scalar('diag/pl_acc_conflict_any_on_u_real', pl_acc_conflict_any_meter.avg, epoch)
            args.writer.add_scalar('diag/pl_acc_conflict_both_on_u_real', pl_acc_conflict_both_meter.avg, epoch)
            args.writer.add_scalar('diag/pl_acc_conflict_any_conf_on_u_real', pl_acc_conflict_any_conf_meter.avg, epoch)
            args.writer.add_scalar('diag/pl_acc_conflict_both_conf_on_u_real', pl_acc_conflict_both_conf_meter.avg, epoch)
            args.writer.add_scalar('diag/L_conflict', L_conflict_meter.avg, epoch)
            args.writer.add_scalar('diag/L_soft', L_soft_meter.avg, epoch)
            args.writer.add_scalar('diag/Lu_h1', Lu_h1_meter.avg, epoch)
            args.writer.add_scalar('diag/Lu_h2', Lu_h2_meter.avg, epoch)
            args.writer.add_scalar('diag/Lu_total', Lu_total_meter.avg, epoch)


            # CSV (master only)
            if args.local_rank in [-1, 0]:
                with open(args.diag_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        epoch,
                        float(getattr(args, 'tau_curr', 0.0)),
                        mask1_meter.avg, mask2_meter.avg, both_conf_meter.avg, disagree_meter.avg, conflict_both_meter.avg,
                        js_w_meter.avg, js_q50, js_q90, js_q99,
                        js_s_meter.avg, agree_h1_ws_meter.avg, agree_h2_ws_meter.avg,
                        pl_acc_h1_meter.avg, pl_acc_h2_meter.avg,
                        pl_acc_h1_conf_meter.avg, pl_acc_h2_conf_meter.avg,
                        conflict_any_meter.avg,
                        pl_acc_conflict_any_meter.avg, pl_acc_conflict_both_meter.avg,
                        pl_acc_conflict_any_conf_meter.avg, pl_acc_conflict_both_conf_meter.avg,
                        L_conflict_meter.avg, L_soft_meter.avg, Lu_h1_meter.avg, Lu_h2_meter.avg, Lu_total_meter.avg,
                    ])

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