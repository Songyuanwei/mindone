import logging
import os
from time import time

from models import VAR, VQVAE, build_vae_var
from utils.arg_util import parse_args
from utils.data import load_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.net_with_loss import GeneratorWithLoss
from utils.utils import load_from_checkpoint

import mindspore as ms
import mindspore.dataset as ds
from mindspore import mint, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import CheckpointManager, resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.recorder import PerfRecorder
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.env import init_train_env
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def create_loss_scaler(loss_scaler_type, init_loss_scale, loss_scale_factor=2, scale_window=1000):
    if loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=init_loss_scale, scale_factor=loss_scale_factor, scale_window=scale_window
        )
    elif loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(init_loss_scale)
    else:
        raise ValueError

    return loss_scaler


def main(args):
    # init
    device_id, rank_id, device_num = init_train_env(
        args.ms_mode,
        seed=args.seed,
        jit_level=args.jit_level,
    )
    set_random_seed(args.seed)
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # dataset
    num_classes, dataset_train, dataset_val = load_dataset(
        args.data_path,
        final_reso=args.data_load_reso,
        hflip=args.hflip,
        mid_reso=args.mid_reso,
    )

    ld_val = ds.GeneratorDataset(
        dataset_val,
        num_workers=0,
        sampler=EvalDistributedSampler(dataset_val, num_replicas=device_num, rank=rank_id),
        shuffle=False,
    )
    ld_val = ld_val.batch(batch_size=round(args.batch_size * 1.5), drop_remainder=False)
    del dataset_val

    ld_train = ds.GeneratorDataset(
        dataset=dataset_train,
        num_workers=args.workers,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train),
            glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True,
            fill_last=True,
            rank=rank_id,
            world_size=device_num,
            start_ep=0,
            start_it=0,
        ),
    )
    del dataset_train
    # iters_train = len(ld_train)
    # ld_train = iter(ld_train)
    dataset_size = ld_train.get_dataset_size()
    logger.info(f"Num batches: {dataset_size}")

    # vae and var
    vae_local, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        patch_nums=args.patch_nums,
        num_classes=num_classes,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
    )
    if args.vae_checkpoint:
        load_from_checkpoint(vae_local, args.vae_checkpoint)
    else:
        print("warning!! VAE uses random initialization!")

    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.dtype in ["fp16", "bf16"]:
        var = auto_mixed_precision(
            var,
            amp_level="O2",
            dtype=dtype_map[args.dtype],
        )

    # build net with loss
    var_with_loss = GeneratorWithLoss(
        patch_nums=args.patch_nums,
        vae_local=vae_local,
        var=var,
        label_smooth=args.ls,
    )

    tot_params, trainable_params = count_params(var_with_loss)
    logger.info("Total params {:,}; Trainable params {:,}".format(tot_params, trainable_params))

    # build lr
    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.scheduler,
        lr=args.base_learning_rate,
        end_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    # build optimizer
    optim_var = create_optimizer(
        var_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    loss_scaler_var = create_loss_scaler(
        args.loss_scaler_type, args.init_loss_scale, args.loss_scale_factor, args.scale_window
    )

    ema = (
        EMA(
            var,
            ema_decay=args.ema_decay,
            offloading=False,
        )
        if args.use_ema
        else None
    )

    # resume training states
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch, cur_iter = 0, 0
    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir, "train_resume.ckpt") if isinstance(args.resume, bool) else args.resume

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            var_with_loss, optim_var, resume_ckpt
        )
        loss_scaler_var.loss_scale_value = loss_scale
        loss_scaler_var.cur_iter = cur_iter
        loss_scaler_var.last_overflow_iter = last_overflow_iter
        logger.info(f"Resume training from {resume_ckpt}")

    training_step_var = TrainOneStepWrapper(
        var_with_loss,
        optimizer=optim_var,
        scale_sense=loss_scaler_var,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )
    if args.train_steps > 0:
        num_epochs = args.train_steps
    else:
        assert args.epochs > 0, "args.epochs must be given and > 0 if train_steps is not specified"
        # the actual data epochs to be run in this case
        num_epochs = args.epochs
    global_step = cur_iter  # index start from 1 (after first-step network update)

    if args.ckpt_save_steps > 0:
        save_by_step = True
    else:
        save_by_step = False
    if rank_id == 0:
        ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        perf_columns = ["step", "loss", "train_time(s)", "shape"]
        output_dir = ckpt_dir.replace("/ckpt", "")
        if start_epoch == 0:
            record = PerfRecorder(output_dir, metric_names=perf_columns)
        else:
            record = PerfRecorder(output_dir, resume=True)

    ds_iter = ld_train.create_tuple_iterator(num_epochs=num_epochs - start_epoch)

    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999.0, 999.0, -1.0, -1.0
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    prog_wp_it = args.pgwp * dataset_size
    L_mean, L_tail = -1, -1
    for epoch in range(start_epoch, num_epochs + 1):
        start_time_e = time.time()
        global_step = epoch * dataset_size
        max_step = num_epochs * dataset_size
        for step, data in enumerate(ds_iter):
            start_time_s = time.time()
            inp = data["image"]
            label = data["label"]
            global_step = global_step + step
            global_step = ms.Tensor(global_step, dtype=ms.int64)
            wp_it = args.wp * dataset_size
            if args.pg:
                if global_step <= wp_it:
                    prog_si = args.pg0
                elif global_step > max_step * args.pg:
                    prog_si = len(args.patch_nums) - 1
                else:
                    delta = len(args.patch_nums) - 1 - args.pg0
                    progress = min(max((global_step - wp_it) / (max_step * args.pg - wp_it), 0), 1)
                    prog_si = args.pg0 + round(progress * delta)
            else:
                prog_si = -1

            loss, overflow, scaling_sens = training_step_var(inp, label, prog_si, prog_wp_it)

            cur_global_step = epoch * dataset_size + step + 1  # starting from 1 for logging
            if overflow:
                logger.warning(f"Overflow occurs in step {cur_global_step}")

            # log
            step_time = time.time() - start_time_s
            if step % args.log_interval == 0:
                loss = float(loss.asnumpy())
                logger.info(f"E: {epoch + 1}, S: {step + 1}, Loss: {loss:.4f}, Step time: {step_time * 1000:.2f}ms")

        epoch_cost = time.time() - start_time_e
        per_step_time = epoch_cost / dataset_size
        cur_epoch = epoch + 1
        logger.info(
            f"Epoch:[{int(cur_epoch):>3d}/{int(args.epochs):>3d}], "
            f"epoch time:{epoch_cost:.2f}s, per step time:{per_step_time * 1000:.2f}ms, "
        )
        if rank_id == 0:
            if (cur_epoch % args.ckpt_save_interval == 0) or (cur_epoch == args.epochs):
                ckpt_name = f"var-e{cur_epoch}.ckpt"
                if ema is not None:
                    ema.swap_before_eval()

                ckpt_manager.save(var, None, ckpt_name=ckpt_name, append_dict=None)
                if ema is not None:
                    ema.swap_after_eval()


if __name__ == "__main__":
    args = parse_args()

    main(args)
