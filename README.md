# Few Pixels, Fragile Faithfulness

![Figure thumbnail](figure/thumnail.png)

## About

This repository implements sparse, prediction-preserving attacks on visual explanations. A genetic algorithm changes a small number of image pixels to shift saliency maps and reduce explanation faithfulness—measured through insertion and deletion objectives—while keeping the model's prediction unchanged. Experiments cover popular vision models and explanation methods, showing that even perturbing only 0.1% of pixels can substantially compromise explanations.

### Quick Start

Run the ResNet18 batch configuration used by `script_sh/resnet18.sh`:

```sh
python run_batch.py \
	--model-name resnet18 \
	--num_sample 100 \
	--operator-strategy uniform \
	--eps 50 \
	--w-del 0.5 \
	--w-ins 0.5 \
	--seed 22520691 \
	--fitness-function margin_loss_causal_faithfull \
	--output-root rivf_official/server_run_seed/GA/22520691/ \
	--explain-method simple_gradient
```

### Single-image attack

```sh
python run_sample.py \
	--image test_imgs/cat.png \
	--output outputs/adv.png \
	--model resnet18 \
	--fitness-function margin_loss_causal_faithfull \
	--w-del 0.5 \
	--w-ins 0.5 \
	--seed 22520691 \
	--save-history-chart
```

Replace the input image path with an existing image. The sample and batch runners
use the same fitness/attacker factories, population defaults (50 individuals,
200 iterations), and per-iteration deletion/insertion logging.

Only `weighted_sum_ga` is supported; NSGA-II and Pareto-front output were removed
from the attack runners. The GA keeps a population for reproduction and **one
best-so-far candidate** across all generations. Only candidates whose predicted
class matches the clean prediction can replace that incumbent, and the candidate
with the smallest weighted sum wins. The incumbent is retained for reproduction,
so a later, worse generation cannot discard it. If no initial candidate preserves
the class, a zero-perturbation clean seed is inserted and reported explicitly.
An explicit `--label` must match the clean prediction.

Margin/CE classification losses are not evaluated or included in the weighted
sum. Existing fitness names remain accepted as compatibility names:

- Saliency variants minimize `w_saliency * saliency_loss` (soft IoU, or negative
  MSE for the MSE variants). Their deletion/insertion values are logged only.
- Causal variants minimize `w_del * del_loss + w_ins * ins_loss`. If either weight
  is explicit, an omitted weight is zero. Otherwise each weight is `w_causal / 2`,
  or `w_saliency / 2` when `--w-causal` is omitted.
- Combined causal+saliency fitness adds `w_saliency * saliency_loss` to that causal
  sum. Explicit deletion/insertion weights take precedence over `--w-causal`.

Batch runners accept `--w-margin 0` for older scripts, but reject nonzero values.
The single-image runner no longer accepts `--w-margin`. Histories contain named
columns with a `#` header and no margin column; summaries expose
`history_weighted_fitness` instead of `history_margin`. Sample charts show weighted
fitness and saliency (or insertion loss for causal-only runs);
`--fitness-chart-output` replaces `--margin-chart-output`.
Historical evaluation scripts for older margin/Pareto results are unchanged.

After each sample attack, the runner re-evaluates insertion/deletion faithfulness
on the clean and final adversarial tensors and saves `<output>_faithfulness.png`.
Both panels compare clean/adversarial curves with their AUC values; these are
causal curves over pixel fractions, not optimization losses over generations.
The RISE evaluator normalizes these curves, so the vertical axis is normalized
score rather than raw probability. Use `--faithfulness-chart-output` to override
the path and `--show-faithfulness-chart` to display the figure interactively
(omit it on headless servers). This uses the in-memory tensors, not the
quantized PNG files. The current RISE evaluator requires CUDA.

Focused regression tests (CPU, no model downloads):

```sh
python -m unittest discover -s tests -p "test_weighted_sum_ga.py"
```

## Citation
If you find our work useful, please cite:
```bibtex
@inproceedings{khoatn-mmm2027,
  title={Few Pixels, Fragile Faithfulness: Evaluating Visual Explanations under Sparse Perturbations},
  author={Khoa Tran, Viet-Hang Duong, Ngoc Hoang Luong},
  booktitle={International Conference on Multimedia Modeling (MM)},
  year={2027},
  publisher = {Springer}
}

@inproceedings{khoatn-mapr2026,
  title     = {On the Instability of Saliency Maps under Sparse Perturbations},
  author    = {Khoa Tran, Thai Huy Nguyen, Quan Minh Phan, Khang Tran, Viet-Hang Duong, Tien-Dung Mai, Ngoc Hoang Luong},
  booktitle = {Proceedings of the 2026 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  year      = {2026},
  publisher = {IEEE}
}
```