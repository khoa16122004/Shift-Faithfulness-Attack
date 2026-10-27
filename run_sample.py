import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.utils import save_image


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from LossFunctions import MarginLossCausalFaithFull
from RISE.evaluation import CausalMetric, auc
from run_batch import (
    _is_saliency_optimized_fitness,
    create_attacker,
    create_fitness,
    save_history_scores_txt,
)
from util import build_blur_substrate, get_explainable_method, get_torchvision_model, save_attack_two_score_charts


class SoftmaxModel(nn.Module):
    def __init__(self, model, dim=1):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, inputs):
        return self.softmax(self.model(inputs))


def parse_args():
    parser = argparse.ArgumentParser(description="Run sparse attack with weighted-sum genetic algorithm")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="adv.png", help="Path to save adversarial image")
    parser.add_argument("--clean-image-output", type=str, default=None, help="Path to save resized clean image")
    parser.add_argument("--clean-map-output", type=str, default=None, help="Path to save clean/original saliency map")
    parser.add_argument("--adv-map-output", type=str, default=None, help="Path to save adversarial saliency map")
    parser.add_argument("--save-history-chart", action="store_true", help="Save separate charts for fitness and saliency scores")
    parser.add_argument("--history-chart-output", type=str, default=None, help="Base path for fitness and saliency chart files")
    parser.add_argument("--fitness-chart-output", type=str, default=None, help="Path to save weighted fitness chart")
    parser.add_argument("--saliency-chart-output", type=str, default=None, help="Path to save saliency score chart")
    parser.add_argument("--faithfulness-chart-output", type=str, default=None, help="Path to save clean/adv insertion and deletion curves (default: <output>_faithfulness.png)")
    parser.add_argument("--show-faithfulness-chart", action="store_true", help="Display re-evaluated insertion/deletion curves after the attack")
    parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing local pretrained checkpoints named <model>.pth",
    )
    parser.add_argument("--label", type=int, default=None, help="True label index. If omitted, uses model prediction")
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
        help="Saliency explanation method",
    )

    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--eps", type=int, default=50, help="Number of perturbed pixels")
    parser.add_argument("--p-size", type=float, default=1.0)
    parser.add_argument("--pc", type=float, default=0.4)
    parser.add_argument("--pm", type=float, default=0.1)
    parser.add_argument("--zero-probability", type=float, default=0.3, help="Probability of zero channel perturbation")
    parser.add_argument("--w-saliency", type=float, default=0.5)
    parser.add_argument("--w-causal", type=float, default=None, help="Weight for averaged causal loss (del+ins)/2 in mixed causal+saliency fitness")
    parser.add_argument("--w-del", type=float, default=None, help="Weight for deletion loss when fitness has 3 objectives")
    parser.add_argument("--w-ins", type=float, default=None, help="Weight for insertion loss when fitness has 3 objectives")
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
            "ce_margin_loss_saliency",
            "CEMarrginLoss",
            "margin_loss_causal_faithfull",
            "ce_loss_causal_faithfull",
            "margin_loss_causal_faithfull_saliency",
        ],
        help="Fitness function to optimize",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="weighted_sum_ga",
        choices=["weighted_sum_ga"],
        help="Optimization algorithm to run",
    )
    parser.add_argument(
        "--operator-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "saliency_guided"],
        help="Population initialization strategy (saliency_guided affects init only)",
    )
    parser.add_argument(
        "--saliency-temperature",
        type=float,
        default=1.0,
        help="Temperature for saliency-guided pixel sampling (lower is sharper)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility")

    return parser.parse_args()



def _save_saliency_map(saliency_map, output_path):
    # Save a single saliency map with a standard "hot" colormap.
    map_2d = saliency_map.detach().float().cpu().squeeze()
    map_min = map_2d.min()
    map_max = map_2d.max()
    den = (map_max - map_min).item()
    if den > 1e-12:
        map_2d = (map_2d - map_min) / (map_max - map_min)
    else:
        map_2d = torch.zeros_like(map_2d)

    # Classic hot map: black -> red -> yellow -> white.
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

    _save_saliency_map(clean_map_a[0], output_paths["clean_a"])
    _save_saliency_map(clean_map_b[0], output_paths["clean_b"])
    _save_saliency_map(adv_map_a[0], output_paths["adv_a"])
    _save_saliency_map(adv_map_b[0], output_paths["adv_b"])


def _compute_faithfulness_metrics(model, normalize, clean_batch, adv_batch, clean_saliency_map, adv_saliency_map):
    metric_model = SoftmaxModel(model)
    blur_fn = build_blur_substrate(11, 5)
    insertion = CausalMetric(metric_model, "ins", 224, substrate_fn=blur_fn)
    deletion = CausalMetric(metric_model, "del", 224, substrate_fn=lambda x: torch.zeros_like(x))

    clean_scores_del = deletion.single_run(
        normalize(clean_batch).detach().cpu(),
        clean_saliency_map.detach().cpu().numpy(),
        verbose=0,
        save_to=None,
    )
    clean_scores_ins = insertion.single_run(
        normalize(clean_batch).detach().cpu(),
        clean_saliency_map.detach().cpu().numpy(),
        verbose=0,
        save_to=None,
    )

    adv_scores_del = deletion.single_run(
        normalize(adv_batch).detach().cpu(),
        adv_saliency_map[0].detach().cpu().numpy(),
        verbose=0,
        save_to=None,
    )
    adv_scores_ins = insertion.single_run(
        normalize(adv_batch).detach().cpu(),
        adv_saliency_map[0].detach().cpu().numpy(),
        verbose=0,
        save_to=None,
    )

    adv_del_auc = float(auc(adv_scores_del))
    adv_ins_auc = float(auc(adv_scores_ins))
    clean_del_auc = float(auc(clean_scores_del))
    clean_ins_auc = float(auc(clean_scores_ins))

    return {
        "clean_del_auc": clean_del_auc,
        "clean_ins_auc": clean_ins_auc,
        "adv_del_auc": adv_del_auc,
        "adv_ins_auc": adv_ins_auc,
        "avg_adv_faithfulness_auc": 0.5 * (adv_del_auc + adv_ins_auc),
        "avg_clean_faithfulness_auc": 0.5 * (clean_del_auc + clean_ins_auc),
        "clean_del_scores": [float(v) for v in clean_scores_del],
        "clean_ins_scores": [float(v) for v in clean_scores_ins],
        "adv_del_scores": [float(v) for v in adv_scores_del],
        "adv_ins_scores": [float(v) for v in adv_scores_ins],
    }


def _save_faithfulness_chart(faithfulness, output_path, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    try:
        for ax, mode, title, xlabel in [
            (axes[0], "del", "Deletion", "Fraction of pixels deleted"),
            (axes[1], "ins", "Insertion", "Fraction of pixels inserted"),
        ]:
            for prefix, label in [("clean", "Clean"), ("adv", "Adversarial")]:
                scores = faithfulness[f"{prefix}_{mode}_scores"]
                auc_value = faithfulness[f"{prefix}_{mode}_auc"]
                ax.plot(
                    np.linspace(0, 1, len(scores)),
                    scores,
                    label=f"{label} (AUC={auc_value:.4f})",
                    linewidth=2,
                )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Normalized score")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("Re-evaluated faithfulness: clean vs adversarial")
        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=180)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def run_attack(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    model, spatial, normalize = get_torchvision_model(
        args.model,
        pretrained=True,
        checkpoint_dir=args.checkpoint_dir,
    )
    model = model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    x_tensor = spatial(image).to(device).unsqueeze(0)

    with torch.no_grad():
        pred = model(normalize(x_tensor)).argmax(dim=1)
    y_true = pred if args.label is None else torch.tensor([args.label], device=device)
    print("Original PRediction: ", pred)

    explain_fn = get_explainable_method(args.explain_method)

    fitness = create_fitness(
        fitness_function=args.fitness_function,
        model=model,
        x_tensor=x_tensor,
        normalize=normalize,
        y_true=y_true,
        explain_fn=explain_fn,
    )

    metric_fitness = None
    if _is_saliency_optimized_fitness(args.fitness_function):
        metric_fitness = MarginLossCausalFaithFull(
            model=model,
            x_tensor=x_tensor,
            normalize=normalize,
            y_true=y_true,
            explain_method=explain_fn,
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
        "w_causal": args.w_causal,
        "w_del": args.w_del,
        "w_ins": args.w_ins,
        "operator_strategy": args.operator_strategy,
        "saliency_temperature": args.saliency_temperature,
        "device": args.device,
        "metric_fitness": metric_fitness,
    }

    attacker = create_attacker(ga_params, args.algorithm)
    adv_chw, best_candidate, best_scores, history = attacker.attack()

    adv_chw = adv_chw.detach().cpu()
    weighted_fitness = best_scores["weighted_fitness"]

    history_del = [float(item["del_loss"]) for item in history if "del_loss" in item]
    history_ins = [float(item["ins_loss"]) for item in history if "ins_loss" in item]
    history_spearman = [float(item["spearman"]) for item in history if "spearman" in item]
    history_saliency = [float(item["saliency_loss"]) for item in history if "saliency_loss" in item]
    history_weighted = [float(item["weighted_fitness"]) for item in history if "weighted_fitness" in item]
    output_root, _ = os.path.splitext(args.output)
    history_scores_path = f"{output_root}_history_scores.txt"
    os.makedirs(os.path.dirname(history_scores_path) or ".", exist_ok=True)
    save_history_scores_txt(history, history_scores_path)

    clean_image_path = args.clean_image_output
    if clean_image_path is None:
        output_root, output_ext = os.path.splitext(args.output)
        output_ext = output_ext if output_ext else ".png"
        clean_image_path = f"{output_root}_clean{output_ext}"

    os.makedirs(os.path.dirname(clean_image_path) or ".", exist_ok=True)
    save_image(x_tensor[0].detach().cpu(), clean_image_path)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_image(adv_chw, args.output)

    with torch.no_grad():
        adv_pred = model(normalize(adv_chw.unsqueeze(0).to(device))).argmax(dim=1).item()

    clean_map_path = args.clean_map_output
    adv_map_path = args.adv_map_output
    if clean_map_path is None or adv_map_path is None:
        output_root, output_ext = os.path.splitext(args.output)
        output_ext = output_ext if output_ext else ".png"
        if clean_map_path is None:
            clean_map_path = f"{output_root}_clean_map{output_ext}"
        if adv_map_path is None:
            adv_map_path = f"{output_root}_adv_map{output_ext}"

    os.makedirs(os.path.dirname(clean_map_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(adv_map_path) or ".", exist_ok=True)

    output_root, output_ext = os.path.splitext(args.output)
    output_ext = output_ext if output_ext else ".png"
    four_map_paths = {
        "clean_a": f"{output_root}_clean_map_class_a{output_ext}",
        "clean_b": f"{output_root}_clean_map_class_b{output_ext}",
        "adv_a": f"{output_root}_adv_map_class_a{output_ext}",
        "adv_b": f"{output_root}_adv_map_class_b{output_ext}",
    }
    for path in four_map_paths.values():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    clean_saliency_map = fitness.saliency_true[0]
    adv_saliency_map, _ = explain_fn(model, adv_chw.unsqueeze(0).to(device), normalize, y_true)
    faithfulness = _compute_faithfulness_metrics(
        model=model,
        normalize=normalize,
        clean_batch=x_tensor,
        adv_batch=adv_chw.unsqueeze(0).to(device),
        clean_saliency_map=clean_saliency_map,
        adv_saliency_map=adv_saliency_map,
    )
    output_root, _ = os.path.splitext(args.output)
    faithfulness_chart_path = args.faithfulness_chart_output or f"{output_root}_faithfulness.png"
    _save_faithfulness_chart(
        faithfulness,
        faithfulness_chart_path,
        show=args.show_faithfulness_chart,
    )
    _save_saliency_map(clean_saliency_map, clean_map_path)
    _save_saliency_map(adv_saliency_map[0], adv_map_path)
    class_a = int(y_true.item())
    class_b = int(adv_pred)
    _save_four_class_maps(
        model=model,
        explain_fn=explain_fn,
        normalize=normalize,
        clean_chw=x_tensor[0],
        adv_chw=adv_chw,
        class_a=class_a,
        class_b=class_b,
        device=device,
        output_paths=four_map_paths,
    )

    fitness_chart_path = None
    saliency_chart_path = None
    should_save_history_chart = (
        args.save_history_chart
        or (args.history_chart_output is not None)
        or (args.fitness_chart_output is not None)
        or (args.saliency_chart_output is not None)
    )
    if should_save_history_chart:
        output_root, output_ext = os.path.splitext(args.output)
        output_ext = output_ext if output_ext else ".png"

        base_root = output_root
        if args.history_chart_output is not None:
            base_root, base_ext = os.path.splitext(args.history_chart_output)
            if base_ext:
                output_ext = base_ext

        fitness_chart_path = args.fitness_chart_output or f"{base_root}_fitness{output_ext}"
        saliency_chart_path = args.saliency_chart_output or f"{base_root}_saliency{output_ext}"

        os.makedirs(os.path.dirname(fitness_chart_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(saliency_chart_path) or ".", exist_ok=True)
        save_attack_two_score_charts(
            history,
            fitness_output_path=fitness_chart_path,
            saliency_output_path=saliency_chart_path,
        )

    print("=== Attack summary ===")
    print(f"model: {args.model}")
    print(f"image: {args.image}")
    print(f"true_label: {y_true}")
    print(f"clean_pred: {pred}")
    print(f"adv_pred: {adv_pred}")
    print(f"l0_distance: {int(best_candidate.l0_distance(adv_chw.to(device)))}")
    if "saliency_loss" in best_scores:
        print(f"saliency_loss: {float(best_scores['saliency_loss'])}")
    if "del_loss" in best_scores and "ins_loss" in best_scores:
        print(f"del_loss: {float(best_scores['del_loss'])}")
        print(f"ins_loss: {float(best_scores['ins_loss'])}")
    print(f"weighted_fitness: {float(weighted_fitness)}")
    print(f"spearman_best_adv_vs_clean: {float(best_scores.get('spearman', float('nan')))}")
    print(f"faithfulness_adv_del_auc: {faithfulness['adv_del_auc']:.6f}")
    print(f"faithfulness_adv_ins_auc: {faithfulness['adv_ins_auc']:.6f}")
    print(f"faithfulness_clean_del_auc: {faithfulness['clean_del_auc']:.6f}")
    print(f"faithfulness_clean_ins_auc: {faithfulness['clean_ins_auc']:.6f}")
    print(f"faithfulness_adv_avg_auc: {faithfulness['avg_adv_faithfulness_auc']:.6f}")
    print(f"first_success_iteration: {best_scores.get('first_success_iteration')}")
    print(f"algorithm: {args.algorithm}")
    print(f"fitness_function: {args.fitness_function}")
    print(f"w_causal: {args.w_causal}")
    print(f"operator_strategy: {args.operator_strategy}")
    print(f"saliency_temperature: {args.saliency_temperature}")
    print(f"seed: {args.seed}")
    print(f"saved_clean_image: {clean_image_path}")
    print(f"saved_adv: {args.output}")
    print(f"saved_faithfulness_chart: {faithfulness_chart_path}")
    print(f"saved_clean_map: {clean_map_path}")
    print(f"saved_adv_map: {adv_map_path}")
    print(f"saved_clean_map_class_a: {four_map_paths['clean_a']}")
    print(f"saved_clean_map_class_b: {four_map_paths['clean_b']}")
    print(f"saved_adv_map_class_a: {four_map_paths['adv_a']}")
    print(f"saved_adv_map_class_b: {four_map_paths['adv_b']}")
    if fitness_chart_path is not None and saliency_chart_path is not None:
        print(f"saved_fitness_chart: {fitness_chart_path}")
        print(f"saved_saliency_chart: {saliency_chart_path}")
    else:
        print("saved_history_chart: disabled (use --save-history-chart)")

    return {
        "model": args.model,
        "image": args.image,
        "true_label": y_true.detach().cpu().tolist(),
        "clean_pred": pred.detach().cpu().tolist(),
        "adv_pred": adv_pred,
        "l0_distance": int(best_candidate.l0_distance(adv_chw.to(device))),
        "saliency_loss": float(best_scores["saliency_loss"]) if "saliency_loss" in best_scores else None,
        "del_loss": float(best_scores["del_loss"]) if "del_loss" in best_scores else None,
        "ins_loss": float(best_scores["ins_loss"]) if "ins_loss" in best_scores else None,
        "weighted_fitness": float(weighted_fitness),
        "spearman_best_adv_vs_clean": float(best_scores.get("spearman", float("nan"))),
        "first_success_iteration": best_scores.get("first_success_iteration"),
        "algorithm": args.algorithm,
        "fitness_function": args.fitness_function,
        "operator_strategy": args.operator_strategy,
        "saliency_temperature": args.saliency_temperature,
        "w_causal": args.w_causal,
        "w_del": args.w_del,
        "w_ins": args.w_ins,
        "seed": args.seed,
        "saved_clean_image": clean_image_path,
        "saved_adv": args.output,
        "saved_faithfulness_chart": faithfulness_chart_path,
        "saved_clean_map": clean_map_path,
        "saved_adv_map": adv_map_path,
        "saved_clean_map_class_a": four_map_paths["clean_a"],
        "saved_clean_map_class_b": four_map_paths["clean_b"],
        "saved_adv_map_class_a": four_map_paths["adv_a"],
        "saved_adv_map_class_b": four_map_paths["adv_b"],
        "saved_fitness_chart": fitness_chart_path,
        "saved_saliency_chart": saliency_chart_path,
        "history_scores_file": history_scores_path,
        "history_weighted_fitness": history_weighted,
        "history_saliency": history_saliency,
        "history_del": history_del,
        "history_ins": history_ins,
        "history_spearman": history_spearman,
        "faithfulness": {
            "del": faithfulness["adv_del_scores"],
            "ins": faithfulness["adv_ins_scores"],
            "clean_del": faithfulness["clean_del_scores"],
            "clean_ins": faithfulness["clean_ins_scores"],
            "auc": {
                "del": faithfulness["adv_del_auc"],
                "ins": faithfulness["adv_ins_auc"],
                "clean_del": faithfulness["clean_del_auc"],
                "clean_ins": faithfulness["clean_ins_auc"],
                "avg_adv": faithfulness["avg_adv_faithfulness_auc"],
                "avg_clean": faithfulness["avg_clean_faithfulness_auc"],
            },
        },
    }


def main():
    args = parse_args()
    run_attack(args)


if __name__ == "__main__":
    main()
