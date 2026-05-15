import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import changedetection.utils_func.lovasz_loss as L
from changedetection.configs.config import get_config
from changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset
from changedetection.models.SCDMoE import SCDMoE
from changedetection.utils_func.mcd_utils import AverageMeter, SCDD_eval_all, accuracy


DATASETS = {
    "SECOND": {
        "dataset_key": "SECOND",
        "num_semantic_classes": 7,
        "train_dataset_path": "/mnt/data1/hq/SECOND/SECOND/SECOND/train",
        "train_data_list_path": "/mnt/data1/hq/SECOND/SECOND/SECOND/train.txt",
        "test_dataset_path": "/mnt/data1/hq/SECOND/SECOND/SECOND/test",
        "test_data_list_path": "/mnt/data1/hq/SECOND/SECOND/SECOND/test.txt",
        "label_cd_divisor": 255.0,
        "batch_size": 2,
        "crop_size": 512,
        "learning_rate": 1e-4,
        "max_iters": 240000,
    },
    "JL1": {
        "dataset_key": "JL1-SCD",
        "num_semantic_classes": 6,
        "train_dataset_path": "/mnt/data1/hq/JL1/JL1/train",
        "train_data_list_path": "/mnt/data1/hq/JL1/JL1/list/train.txt",
        "test_dataset_path": "/mnt/data1/hq/JL1/JL1/test",
        "test_data_list_path": "/mnt/data1/hq/JL1/JL1/list/test.txt",
        "label_cd_divisor": 1.0,
        "batch_size": 12,
        "crop_size": 256,
        "learning_rate": 2e-4,
        "max_iters": 960000,
    },
    "Landsat": {
        "dataset_key": "Landsat",
        "num_semantic_classes": 5,
        "train_dataset_path": "/mnt/data1/hq/Landsat/Landsat-SCD",
        "train_data_list_path": "/mnt/data1/hq/Landsat/Landsat-SCD/train_list_old.txt",
        "test_dataset_path": "/mnt/data1/hq/Landsat/Landsat-SCD",
        "test_data_list_path": "/mnt/data1/hq/Landsat/Landsat-SCD/test_list_old.txt",
        "label_cd_divisor": 1.0,
        "batch_size": 8,
        "crop_size": 416,
        "learning_rate": 1e-4,
        "max_iters": 960000,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def expand_list(names, max_iters):
    if max_iters <= 0:
        return names
    repeats = int(np.ceil(float(max_iters) / len(names)))
    return (names * repeats)[:max_iters]


def build_model(args):
    config = get_config(args)
    model = SCDMoE(
        output_cd=2,
        output_clf=args.num_semantic_classes,
        pretrained=args.pretrained_weight_path,
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        state_dict = model.state_dict()
        matched = {k: v for k, v in checkpoint.items() if k in state_dict and state_dict[k].shape == v.shape}
        state_dict.update(matched)
        model.load_state_dict(state_dict)
        print(f"Resumed {len(matched)}/{len(state_dict)} tensors from {args.resume}")

    return model.cuda()


def make_train_loader(args):
    names = expand_list(read_list(args.train_data_list_path), args.max_iters)
    dataset = SemanticChangeDetectionDatset(args.train_dataset_path, names, args.crop_size, None, "train")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def make_val_loader(args):
    names = read_list(args.test_data_list_path)
    if args.max_val_images > 0:
        names = names[: args.max_val_images]
    dataset = SemanticChangeDetectionDatset(args.test_dataset_path, names, args.crop_size, None, "test")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)


def compute_loss(output_cd, output_a, output_b, labels_cd, labels_a, labels_b, kl_loss):
    ce_loss_cd = F.cross_entropy(output_cd, labels_cd)
    lovasz_loss_cd = L.lovasz_softmax(F.softmax(output_cd, dim=1), labels_cd)

    ce_loss_a = F.cross_entropy(output_a, labels_a)
    lovasz_loss_a = L.lovasz_softmax(F.softmax(output_a, dim=1), labels_a)

    ce_loss_b = F.cross_entropy(output_b, labels_b)
    lovasz_loss_b = L.lovasz_softmax(F.softmax(output_b, dim=1), labels_b)

    similarity_mask = (labels_a == 0).float().unsqueeze(1).expand_as(output_a)
    similarity_loss = F.mse_loss(
        F.softmax(output_a, dim=1) * similarity_mask,
        F.softmax(output_b, dim=1) * similarity_mask,
        reduction="mean",
    )

    total = (
        ce_loss_cd
        + 0.5 * (ce_loss_a + ce_loss_b + 0.5 * similarity_loss)
        + 0.75 * (lovasz_loss_cd + 0.5 * (lovasz_loss_a + lovasz_loss_b))
        + kl_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "cd": float((ce_loss_cd + lovasz_loss_cd).detach().cpu()),
        "sem": float(((ce_loss_a + ce_loss_b + lovasz_loss_a + lovasz_loss_b) / 2).detach().cpu()),
        "kl": float(kl_loss.detach().cpu() if torch.is_tensor(kl_loss) else kl_loss),
    }


def validate(model, args):
    model.eval()
    num_land_classes = args.num_semantic_classes - 1
    num_scd_classes = num_land_classes * num_land_classes + 1
    loader = make_val_loader(args)
    acc_meter = AverageMeter()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for data in loader:
            pre, post, labels_cd, labels_a, labels_b, _ = data
            pre = pre.cuda(non_blocking=True)
            post = post.cuda(non_blocking=True)
            labels_cd_tensor = (labels_cd / args.label_cd_divisor).cuda(non_blocking=True).long()
            labels_a_tensor = labels_a.cuda(non_blocking=True).long()
            labels_b_tensor = labels_b.cuda(non_blocking=True).long()

            output_cd, output_a, output_b, _ = model(pre, post, False)
            change_mask = torch.argmax(output_cd, dim=1).cpu().numpy()
            preds_a = torch.argmax(output_a, dim=1).cpu().numpy()
            preds_b = torch.argmax(output_b, dim=1).cpu().numpy()
            labels_cd_np = labels_cd_tensor.cpu().numpy()
            labels_a_np = labels_a_tensor.cpu().numpy()
            labels_b_np = labels_b_tensor.cpu().numpy()

            preds_scd = (preds_a - 1) * num_land_classes + preds_b
            preds_scd[change_mask == 0] = 0
            labels_scd = (labels_a_np - 1) * num_land_classes + labels_b_np
            labels_scd[labels_cd_np == 0] = 0

            for pred_scd, label_scd in zip(preds_scd, labels_scd):
                acc, _ = accuracy(pred_scd, label_scd)
                acc_meter.update(acc)
                preds_all.append(pred_scd)
                labels_all.append(label_scd)

    kappa, fscd, miou, sek = SCDD_eval_all(preds_all, labels_all, num_scd_classes)
    metrics = {
        "kappa": float(kappa),
        "Fscd": float(fscd),
        "OA": float(acc_meter.avg),
        "mIoU": float(miou),
        "SeK": float(sek),
    }
    print(
        f"Validation: Kappa={metrics['kappa']:.6f}, Fscd={metrics['Fscd']:.6f}, "
        f"OA={metrics['OA']:.6f}, mIoU={metrics['mIoU']:.6f}, SeK={metrics['SeK']:.6f}"
    )
    model.train()
    return metrics


def write_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args):
    set_seed(args.seed)
    model = build_model(args)
    train_loader = make_train_loader(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    run_dir = Path(args.output_dir) / args.dataset / f"{args.run_name}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    best_sek = -1e9
    best_metrics = None

    print(f"Saving checkpoints to {run_dir}")
    model.train()
    for itera, data in enumerate(tqdm(train_loader, desc=f"Train {args.dataset}"), start=1):
        pre, post, labels_cd, labels_a, labels_b, _ = data
        pre = pre.cuda(non_blocking=True)
        post = post.cuda(non_blocking=True)
        labels_cd = (labels_cd / args.label_cd_divisor).cuda(non_blocking=True).long()
        labels_a = labels_a.cuda(non_blocking=True).long()
        labels_b = labels_b.cuda(non_blocking=True).long()

        output_cd, output_a, output_b, kl_loss = model(pre, post)
        optimizer.zero_grad()
        loss, loss_items = compute_loss(output_cd, output_a, output_b, labels_cd, labels_a, labels_b, kl_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if itera % args.log_interval == 0:
            print(
                f"iter={itera}, loss={loss_items['loss']:.6f}, cd={loss_items['cd']:.6f}, "
                f"sem={loss_items['sem']:.6f}, kl={loss_items['kl']:.6f}, lr={scheduler.get_last_lr()[0]:.8f}"
            )

        if args.val_interval > 0 and itera >= args.val_start and itera % args.val_interval == 0:
            metrics = validate(model, args)
            row = {"iter": itera, **loss_items, **metrics}
            write_log(log_path, row)
            if metrics["SeK"] > best_sek:
                best_sek = metrics["SeK"]
                best_metrics = metrics
                ckpt_path = run_dir / f"{itera}_model_{best_sek}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved best checkpoint to {ckpt_path}")

        if args.save_interval > 0 and itera % args.save_interval == 0:
            ckpt_path = run_dir / f"{itera}_latest.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved periodic checkpoint to {ckpt_path}")

    if best_metrics is not None:
        print(f"Best metrics: {best_metrics}")
    else:
        ckpt_path = run_dir / "final_model.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"No validation was run; saved final checkpoint to {ckpt_path}")


def apply_dataset_defaults(args):
    spec = DATASETS[args.dataset]
    for key, value in spec.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.dataset_key is None:
        args.dataset_key = spec["dataset_key"]
    args.dataset_for_loader = args.dataset_key
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Train SCD-MoE.")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    parser.add_argument("--cfg", default=str(ROOT / "changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"))
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--pretrained_weight_path", default=str(ROOT / "pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dataset_key", default=None)
    parser.add_argument("--num_semantic_classes", type=int, default=None)
    parser.add_argument("--train_dataset_path", default=None)
    parser.add_argument("--train_data_list_path", default=None)
    parser.add_argument("--test_dataset_path", default=None)
    parser.add_argument("--test_data_list_path", default=None)
    parser.add_argument("--label_cd_divisor", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_step_size", type=int, default=10000)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--val_start", type=int, default=500)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--output_dir", default=str(ROOT / "saved_models"))
    parser.add_argument("--run_name", default="SCD-MoE")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    args = apply_dataset_defaults(args)
    # Compatibility with config/data-loader code that expects args.dataset.
    args.dataset = args.dataset_key
    return args


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
