import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from changedetection.configs.config import get_config
from changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset
from changedetection.models.SCDMoE import SCDMoE
from changedetection.utils_func.mcd_utils import AverageMeter, SCDD_eval_all, accuracy


DATASETS = {
    "SECOND": {
        "num_semantic_classes": 7,
        "test_dataset_path": "/mnt/data1/hq/SECOND/SECOND/SECOND/test",
        "test_data_list_path": "/mnt/data1/hq/SECOND/SECOND/SECOND/test.txt",
        "resume": str(ROOT / "checkpoints/SECOND/scd_moe_second.pth"),
        "label_cd_divisor": 255.0,
    },
    "JL1": {
        "num_semantic_classes": 6,
        "test_dataset_path": "/mnt/data1/hq/JL1/JL1/test",
        "test_data_list_path": "/mnt/data1/hq/JL1/JL1/list/test.txt",
        "resume": str(ROOT / "checkpoints/JL1/scd_moe_jl1.pth"),
        "label_cd_divisor": 1.0,
    },
    "Landsat": {
        "num_semantic_classes": 5,
        "test_dataset_path": "/mnt/data1/hq/Landsat/Landsat-SCD",
        "test_data_list_path": "/mnt/data1/hq/Landsat/Landsat-SCD/test_list_old.txt",
        "resume": str(ROOT / "checkpoints/Landsat/scd_moe_landsat.pth"),
        "label_cd_divisor": 1.0,
    },
}


def build_model(args, num_semantic_classes):
    config = get_config(args)
    model = SCDMoE(
        output_cd=2,
        output_clf=num_semantic_classes,
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

    checkpoint = torch.load(args.resume, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    state_dict = model.state_dict()
    matched = {k: v for k, v in checkpoint.items() if k in state_dict and state_dict[k].shape == v.shape}
    state_dict.update(matched)
    model.load_state_dict(state_dict)
    print(f"Loaded {len(matched)}/{len(state_dict)} tensors from {args.resume}")
    return model.cuda().eval()


def infer_one(args):
    spec = DATASETS[args.dataset]
    num_semantic_classes = int(args.num_semantic_classes or spec["num_semantic_classes"])
    num_land_classes = num_semantic_classes - 1
    num_scd_classes = num_land_classes * num_land_classes + 1
    test_dataset_path = args.test_dataset_path or spec["test_dataset_path"]
    test_data_list_path = args.test_data_list_path or spec["test_data_list_path"]
    label_cd_divisor = float(args.label_cd_divisor or spec["label_cd_divisor"])

    with open(test_data_list_path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    if args.max_images > 0:
        names = names[: args.max_images]

    dataset = SemanticChangeDetectionDatset(test_dataset_path, names, args.crop_size, None, "test")
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, drop_last=False)
    model = build_model(args, num_semantic_classes)

    acc_meter = AverageMeter()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for idx, data in enumerate(loader):
            pre, post, labels_cd, labels_a, labels_b, _ = data
            pre = pre.cuda(non_blocking=True)
            post = post.cuda(non_blocking=True)
            labels_cd_tensor = (labels_cd / label_cd_divisor).cuda(non_blocking=True).long()
            labels_a_tensor = labels_a.cuda(non_blocking=True).long()
            labels_b_tensor = labels_b.cuda(non_blocking=True).long()

            output_cd, output_a, output_b, _ = model(pre, post)
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

            if args.print_freq > 0 and (idx + 1) % args.print_freq == 0:
                print(f"[{args.dataset}] processed {idx + 1}/{len(loader)}")

    kappa_n0, fscd, miou, sek = SCDD_eval_all(preds_all, labels_all, num_scd_classes)
    metrics = {
        "dataset": args.dataset,
        "checkpoint": args.resume,
        "num_images": len(names),
        "kappa": float(kappa_n0),
        "Fscd": float(fscd),
        "OA": float(acc_meter.avg),
        "mIoU": float(miou),
        "SeK": float(sek),
    }
    print(
        f"{args.dataset}: Kappa={metrics['kappa']:.6f}, Fscd={metrics['Fscd']:.6f}, "
        f"OA={metrics['OA']:.6f}, mIoU={metrics['mIoU']:.6f}, SeK={metrics['SeK']:.6f}"
    )
    return metrics


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate packaged SCD-MoE checkpoints.")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    parser.add_argument("--cfg", default=str(ROOT / "changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"))
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--pretrained_weight_path", default=str(ROOT / "pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--test_dataset_path", default=None)
    parser.add_argument("--test_data_list_path", default=None)
    parser.add_argument("--num_semantic_classes", type=int, default=None)
    parser.add_argument("--label_cd_divisor", type=float, default=None)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=200)
    parser.add_argument("--output_csv", default=str(ROOT / "results/eval_metrics.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    rows = []
    for dataset_name in datasets:
        run_args = argparse.Namespace(**vars(args))
        run_args.dataset = dataset_name
        if args.resume is None:
            run_args.resume = DATASETS[dataset_name]["resume"]
        rows.append(infer_one(run_args))
    write_metrics(Path(args.output_csv), rows)
    print(f"Saved metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
