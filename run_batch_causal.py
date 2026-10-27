import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.utils import save_image
from tqdm.auto import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from LossFunctions import (
    MarginLosssMSESaliency_Fitness,
    MarginSalinecy_Fitness,
    NegativeCrossEntropySaliency_Fitness,
    ReverseMarginSalinecy_Fitness,
    ReverseNegativeCrossEntropySaliency_Fitness,
)
from run_batch import create_attacker, save_history_scores_txt
from RISE.evaluation import CausalMetric, auc
from imagenet_metadata import IMAGENET_CLASSNAMES
from util import (
    build_blur_substrate,
    get_explainable_method,
    get_torchvision_model,
)


DEFAULT_IMAGENET_VAL_ROOT = r"E:\ImageNet1K\imagenet\ImageNet1K\val"
DEFAULT_REMOTE_VAL_ROOT = "/datastore/elo/quanphm/dataset/ImageNet1K/val"


class SoftmaxModel(nn.Module):
    def __init__(self, model, dim=1):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, inputs):
        return self.softmax(self.model(inputs))


def parse_args():
    parser = argparse.ArgumentParser(description="Batch attack + causal metrics runner")
    parser.add_argument("--selection-file", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--num_sample", type=int, default=None)
    parser.add_argument("--imagenet-val-root", type=str, default=DEFAULT_IMAGENET_VAL_ROOT)
    parser.add_argument("--replace-from-root", type=str, default=DEFAULT_REMOTE_VAL_ROOT)
    parser.add_argument("--output-root", type=str, default="batch_outputs_causal")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing local pretrained checkpoints named <model>.pth",
    )
    parser.add_argument("--replace", action="store_true")

    parser.add_argument(
        "--explain-method",
        type=str,
        default="simple_gradient",
        choices=[
            "simple_gradient",
            "integrated_gradients",
            "input_gradient",
            "grad_cam",
            "grad_cam_plusplus",
            "xgrad_cam",
            "layer_cam",
            "hires_cam",
            "score_cam",
            "ablation_cam",
            "eigen_cam",
            "eigen_grad_cam",
            "fullgrad",
            "raw_attention",
            "attention_grad",
        ],
    )
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--eps", type=int, default=50)
    parser.add_argument("--p-size", type=float, default=1.0)
    parser.add_argument("--pc", type=float, default=0.4)
    parser.add_argument("--pm", type=float, default=0.1)
    parser.add_argument("--zero-probability", type=float, default=0.3)
    parser.add_argument("--w-margin", type=float, default=0.0, help="Deprecated compatibility option; must be 0")
    parser.add_argument("--w-saliency", type=float, default=0.5)
    parser.add_argument(
        "--fitness-function",
        type=str,
        default="margin_saliency",
        choices=[
            "margin_saliency",
            "negative_cross_entropy_saliency",
            "cross_entropy_saliency",
            "reverse_margin_saliency",
            "reverse_negative_cross_entropy_saliency",
            "margin_loss_mse_saliency",
            "MarginLossMESE",
        ],
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="weighted_sum_ga",
        choices=["weighted_sum_ga"],
    )
    parser.add_argument(
        "--operator-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "saliency_guided"],
    )
    parser.add_argument("--saliency-temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=int, default=5)
    parser.add_argument("--save-process", action="store_true")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])

    args = parser.parse_args()
    if args.w_margin != 0:
        parser.error("--w-margin must be 0; margin loss was removed")
    return args


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _fmt_num(value):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}".replace("+", "")
    return str(value)


def _save_saliency_map(saliency_map, output_path):
    map_2d = saliency_map.detach().float().cpu().squeeze()
    map_min = map_2d.min()
    map_max = map_2d.max()
    den = (map_max - map_min).item()

    if den > 1e-12:
        map_2d = (map_2d - map_min) / (map_max - map_min)
    else:
        map_2d = torch.zeros_like(map_2d)

    r = torch.clamp(3.0 * map_2d, 0.0, 1.0)
    g = torch.clamp(3.0 * map_2d - 1.0, 0.0, 1.0)
    b = torch.clamp(3.0 * map_2d - 2.0, 0.0, 1.0)
    rgb = (torch.stack([r, g, b], dim=-1) * 255.0).clamp(0, 255).byte().numpy()
    Image.fromarray(rgb, mode="RGB").save(output_path)


def _save_four_class_maps(model, explain_fn, normalize, clean_chw, adv_chw, class_a, class_b, device, output_paths):
    clean_batch = clean_chw.unsqueeze(0).to(device)
    adv_batch = adv_chw.unsqueeze(0).to(device)

    target_a = torch.tensor([int(class_a)], device=device)
    target_b = torch.tensor([int(class_b)], device=device)

    clean_map_a, _ = explain_fn(model, clean_batch, normalize, target_a)
    clean_map_b, _ = explain_fn(model, clean_batch, normalize, target_b)
    adv_map_a, _ = explain_fn(model, adv_batch, normalize, target_a)
    adv_map_b, _ = explain_fn(model, adv_batch, normalize, target_b)

    _save_saliency_map(clean_map_a[0], str(output_paths["clean_a"]))
    _save_saliency_map(clean_map_b[0], str(output_paths["clean_b"]))
    _save_saliency_map(adv_map_a[0], str(output_paths["adv_a"]))
    _save_saliency_map(adv_map_b[0], str(output_paths["adv_b"]))


def history_to_lists(history):
    saliency = []
    for item in history:
        saliency.append(_to_float(item["saliency_loss"]))
    return saliency


def load_selection_file(selection_file):
    with open(selection_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Selection file must contain a JSON object: {selection_file}")
    return data


def infer_model_name_from_selection_file(selection_file):
    stem = Path(selection_file).stem
    suffix = "_selection"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def resolve_image_path(raw_path, class_name, imagenet_val_root, replace_from_root):
    image_path = Path(raw_path)
    if image_path.exists():
        return image_path

    normalized_raw = str(raw_path).replace("\\", "/")
    normalized_old_root = replace_from_root.replace("\\", "/").rstrip("/")

    if normalized_raw.startswith(normalized_old_root + "/"):
        suffix = normalized_raw[len(normalized_old_root) + 1 :]
        candidate = Path(imagenet_val_root) / Path(suffix)
        if candidate.exists():
            return candidate

    image_name = Path(raw_path).name
    return Path(imagenet_val_root) / class_name / image_name


def prepare_output_paths(output_dir):
    return {
        "adv": output_dir / "adv.png",
        "clean": output_dir / "clean.png",
        "clean_map": output_dir / "clean_map.png",
        "adv_map": output_dir / "adv_map.png",
        "clean_map_class_a": output_dir / "clean_map_class_a.png",
        "clean_map_class_b": output_dir / "clean_map_class_b.png",
        "adv_map_class_a": output_dir / "adv_map_class_a.png",
        "adv_map_class_b": output_dir / "adv_map_class_b.png",
        "history_txt": output_dir / "history_scores.txt",
        "summary": output_dir / "summary.json",
    }


def build_approach_tag(args):
    parts = [
        f"strategy-{args.operator_strategy}",
        f"ws-{_fmt_num(args.w_saliency)}",
        f"eps-{_fmt_num(args.eps)}",
        f"ps-{_fmt_num(args.p_size)}",
        f"zp-{_fmt_num(args.zero_probability)}",
        f"temp-{_fmt_num(args.saliency_temperature)}",
        f"exp-{args.explain_method}",
    ]
    if args.fitness_function != "margin_saliency":
        fit_name = args.fitness_function
        if fit_name == "cross_entropy_saliency":
            fit_name = "negative_cross_entropy_saliency"
        if fit_name == "MarginLossMESE":
            fit_name = "margin_loss_mse_saliency"
        parts.append(f"fit-{fit_name}")
    if args.seed is not None:
        parts.append(f"seed-{args.seed}")
    return "__".join(parts)


def create_fitness(fitness_function, model, x_tensor, normalize, y_true, explain_fn):
    if fitness_function == "margin_saliency":
        return MarginSalinecy_Fitness(model=model, x_tensor=x_tensor, normalize=normalize, y_true=y_true, explain_method=explain_fn)
    if fitness_function in {"negative_cross_entropy_saliency", "cross_entropy_saliency"}:
        return NegativeCrossEntropySaliency_Fitness(model=model, x_tensor=x_tensor, normalize=normalize, y_true=y_true, explain_method=explain_fn)
    if fitness_function == "reverse_margin_saliency":
        return ReverseMarginSalinecy_Fitness(model=model, x_tensor=x_tensor, normalize=normalize, y_true=y_true, explain_method=explain_fn)
    if fitness_function == "reverse_negative_cross_entropy_saliency":
        return ReverseNegativeCrossEntropySaliency_Fitness(model=model, x_tensor=x_tensor, normalize=normalize, y_true=y_true, explain_method=explain_fn)
    if fitness_function in {"margin_loss_mse_saliency", "MarginLossMESE"}:
        return MarginLosssMSESaliency_Fitness(model=model, x_tensor=x_tensor, normalize=normalize, y_true=y_true, explain_method=explain_fn)
    raise ValueError(f"Unsupported fitness function: {fitness_function}")


def _compute_causal_metrics(model, normalize, clean_batch, adv_batch, clean_saliency_map, adv_saliency_map, output_dir, args):
    metric_model = SoftmaxModel(model)

    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    insertion = CausalMetric(metric_model, "ins", args.step, substrate_fn=blur_fn)
    deletion = CausalMetric(metric_model, "del", args.step, substrate_fn=lambda x: torch.zeros_like(x))

    clean_del_steps_dir = output_dir / "clean_deletion_steps"
    clean_ins_steps_dir = output_dir / "clean_insertion_steps"
    adv_del_steps_dir = output_dir / "adv_deletion_steps"
    adv_ins_steps_dir = output_dir / "adv_insertion_steps"

    if args.save_process:
        clean_del_steps_dir.mkdir(parents=True, exist_ok=True)
        clean_ins_steps_dir.mkdir(parents=True, exist_ok=True)
        adv_del_steps_dir.mkdir(parents=True, exist_ok=True)
        adv_ins_steps_dir.mkdir(parents=True, exist_ok=True)

    clean_scores_del = deletion.single_run(
        normalize(clean_batch).cpu().detach(),
        clean_saliency_map.cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(clean_del_steps_dir) if args.save_process else None,
    )
    clean_scores_ins = insertion.single_run(
        normalize(clean_batch).cpu().detach(),
        clean_saliency_map.cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(clean_ins_steps_dir) if args.save_process else None,
    )

    adv_scores_del = deletion.single_run(
        normalize(adv_batch).cpu().detach(),
        clean_saliency_map.cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(adv_del_steps_dir) if args.save_process else None,
    )
    adv_scores_ins = insertion.single_run(
        normalize(adv_batch).cpu().detach(),
        adv_saliency_map[0].cpu().detach().numpy(),
        verbose=args.verbose,
        save_to=str(adv_ins_steps_dir) if args.save_process else None,
    )

    return {
        "clean_del_auc": float(auc(clean_scores_del)),
        "clean_ins_auc": float(auc(clean_scores_ins)),
        "adv_del_auc": float(auc(adv_scores_del)),
        "adv_ins_auc": float(auc(adv_scores_ins)),
        "clean_del_scores": [float(v) for v in clean_scores_del],
        "clean_ins_scores": [float(v) for v in clean_scores_ins],
        "adv_del_scores": [float(v) for v in adv_scores_del],
        "adv_ins_scores": [float(v) for v in adv_scores_ins],
    }


def run_attack_one(image_path, output_paths, model_name, model, spatial, normalize, explain_fn, args, device, sample_seed=None):
    if sample_seed is not None:
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

    image = Image.open(image_path).convert("RGB")
    x_tensor = spatial(image).to(device).unsqueeze(0)

    with torch.no_grad():
        pred = model(normalize(x_tensor)).argmax(dim=1)
    y_true = pred if args.label is None else torch.tensor([args.label], device=device)

    fitness = create_fitness(
        fitness_function=args.fitness_function,
        model=model,
        x_tensor=x_tensor,
        normalize=normalize,
        y_true=y_true,
        explain_fn=explain_fn,
    )

    ga_params = {
        "x_tensor": x_tensor,
        "normalize": normalize,
        "fitness": fitness,
        "pop_size": args.pop_size,
        "iterations": args.iterations,
        "eps": args.eps,
        "p_size": args.p_size,
        "pc": args.pc,
        "pm": args.pm,
        "zero_probability": args.zero_probability,
        "all_pixels": torch.arange(x_tensor.shape[-2] * x_tensor.shape[-1], device=device),
        "w_saliency": args.w_saliency,
        "operator_strategy": args.operator_strategy,
        "saliency_temperature": args.saliency_temperature,
        "device": args.device,
    }

    attacker = create_attacker(ga_params, args.algorithm)
    adv_chw, best_candidate, best_scores, history = attacker.attack()

    adv_chw_cpu = adv_chw.detach().cpu()
    save_image(x_tensor[0].detach().cpu(), str(output_paths["clean"]))
    save_image(adv_chw_cpu, str(output_paths["adv"]))

    with torch.no_grad():
        adv_pred = model(normalize(adv_chw.unsqueeze(0).to(device))).argmax(dim=1).item()

    clean_pred = int(pred.item())
    class_name = IMAGENET_CLASSNAMES[clean_pred] if 0 <= clean_pred < len(IMAGENET_CLASSNAMES) else str(clean_pred)

    clean_saliency_map = fitness.saliency_true[0]
    adv_saliency_map, _ = explain_fn(model, adv_chw.unsqueeze(0).to(device), normalize, y_true)

    _save_saliency_map(clean_saliency_map, str(output_paths["clean_map"]))
    _save_saliency_map(adv_saliency_map[0], str(output_paths["adv_map"]))
    _save_four_class_maps(
        model=model,
        explain_fn=explain_fn,
        normalize=normalize,
        clean_chw=x_tensor[0],
        adv_chw=adv_chw,
        class_a=int(y_true.item()),
        class_b=int(adv_pred),
        device=device,
        output_paths={
            "clean_a": output_paths["clean_map_class_a"],
            "clean_b": output_paths["clean_map_class_b"],
            "adv_a": output_paths["adv_map_class_a"],
            "adv_b": output_paths["adv_map_class_b"],
        },
    )

    save_history_scores_txt(history, str(output_paths["history_txt"]))

    weighted_fitness = best_scores["weighted_fitness"]

    causal = _compute_causal_metrics(
        model=model,
        normalize=normalize,
        clean_batch=x_tensor,
        adv_batch=adv_chw_cpu.unsqueeze(0),
        clean_saliency_map=clean_saliency_map,
        adv_saliency_map=adv_saliency_map,
        output_dir=output_paths["summary"].parent,
        args=args,
    )

    history_saliency = history_to_lists(history)

    return {
        "model": model_name,
        "seed": sample_seed,
        "true_label": int(y_true.item()),
        "clean_pred": clean_pred,
        "adv_pred": int(adv_pred),
        "class_name": class_name,
        "l0_distance": int(best_candidate.l0_distance(adv_chw_cpu.to(device))),
        "saliency_loss": float(best_scores["saliency_loss"]),
        "weighted_fitness": float(weighted_fitness),
        "first_success_iteration": best_scores.get("first_success_iteration"),
        "algorithm": args.algorithm,
        "fitness_function": args.fitness_function,
        "operator_strategy": args.operator_strategy,
        "saliency_temperature": float(args.saliency_temperature),
        "history_scores_file": str(output_paths["history_txt"]),
        "history_weighted_fitness": [float(item["weighted_fitness"]) for item in history],
        "history_saliency": history_saliency,
        "causual": {
            "del": causal["adv_del_scores"],
            "ins": causal["adv_ins_scores"],
            "clean_del": causal["clean_del_scores"],
            "clean_ins": causal["clean_ins_scores"],
            "auc": {
                "del": causal["adv_del_auc"],
                "ins": causal["adv_ins_auc"],
                "clean_del": causal["clean_del_auc"],
                "clean_ins": causal["clean_ins_auc"],
            },
        },
    }


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.num_sample is not None and args.num_sample < 0:
        raise ValueError("--num_sample must be >= 0")

    if args.selection_file is None:
        if not args.model_name:
            raise ValueError("Provide --model-name or --selection-file")
        selection_file = Path("model_evaluation_results") / f"{args.model_name}_selection_random50.json"
    else:
        selection_file = Path(args.selection_file)

    if not selection_file.exists():
        raise FileNotFoundError(f"Selection file not found: {selection_file}")

    model_name = args.model_name or infer_model_name_from_selection_file(selection_file)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    print(f"[INFO] selection_file={selection_file}")
    print(f"[INFO] model={model_name}")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] seed={args.seed}")

    model, spatial, normalize = get_torchvision_model(
        model_name,
        pretrained=True,
        checkpoint_dir=args.checkpoint_dir,
    )
    model = model.to(device)
    model.eval()

    explain_fn = get_explainable_method(args.explain_method)

    selections = load_selection_file(selection_file)
    items = list(selections.items())
    if args.num_sample is not None:
        items = items[: args.num_sample]

    approach_tag = build_approach_tag(args)
    run_root = Path(args.output_root) / model_name / approach_tag
    run_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_ok = 0
    total_failed = 0
    total_skipped = 0
    total_missing = 0

    progress = tqdm(items, total=len(items), desc="Running causal batch", unit="img")

    for sample_index, (class_name, raw_path) in enumerate(progress):
        image_path = resolve_image_path(
            raw_path=raw_path,
            class_name=class_name,
            imagenet_val_root=args.imagenet_val_root,
            replace_from_root=args.replace_from_root,
        )

        image_stem = image_path.stem
        output_dir = run_root / class_name / image_stem
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = prepare_output_paths(output_dir)

        if output_paths["summary"].exists() and not args.replace:
            result = {
                "status": "skipped",
                "reason": "exists",
                "model": model_name,
                "class": class_name,
                "input_raw": raw_path,
                "resolved_image": str(image_path),
                "output_dir": str(output_dir),
            }
            all_results.append(result)
            total_skipped += 1
            continue

        if not image_path.exists():
            result = {
                "status": "missing_image",
                "model": model_name,
                "class": class_name,
                "input_raw": raw_path,
                "resolved_image": str(image_path),
                "output_dir": str(output_dir),
            }
            with open(output_paths["summary"], "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            all_results.append(result)
            total_missing += 1
            continue

        try:
            sample_seed = None if args.seed is None else (args.seed + sample_index)
            metrics = run_attack_one(
                image_path=image_path,
                output_paths=output_paths,
                model_name=model_name,
                model=model,
                spatial=spatial,
                normalize=normalize,
                explain_fn=explain_fn,
                args=args,
                device=device,
                sample_seed=sample_seed,
            )
            result = {
                "status": "ok",
                "class": class_name,
                "input_raw": raw_path,
                "resolved_image": str(image_path),
                "output_dir": str(output_dir),
            }
            result.update(metrics)
            total_ok += 1
        except Exception as exc:
            result = {
                "status": "failed",
                "model": model_name,
                "class": class_name,
                "input_raw": raw_path,
                "resolved_image": str(image_path),
                "output_dir": str(output_dir),
                "error": str(exc),
            }
            total_failed += 1

        with open(output_paths["summary"], "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        all_results.append(result)

    report = {
        "selection_file": str(selection_file),
        "model": model_name,
        "approach": approach_tag,
        "seed": args.seed,
        "num_requested": args.num_sample,
        "total": len(all_results),
        "ok": total_ok,
        "failed": total_failed,
        "missing_image": total_missing,
        "skipped": total_skipped,
        "imagenet_val_root": args.imagenet_val_root,
        "replace_from_root": args.replace_from_root,
        "results": all_results,
    }

    report_path = run_root / "batch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Batch summary ===")
    print(f"report: {report_path}")
    print(f"total: {report['total']}")
    print(f"ok: {report['ok']}")
    print(f"failed: {report['failed']}")
    print(f"missing_image: {report['missing_image']}")
    print(f"skipped: {report['skipped']}")


if __name__ == "__main__":
    main()
