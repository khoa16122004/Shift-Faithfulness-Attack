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
	--w-margin 0.0 \
	--w-del 0.5 \
	--w-ins 0.5 \
	--seed 22520691 \
	--fitness-function margin_loss_causal_faithfull \
	--output-root rivf_official/server_run_seed/GA/22520691/ \
	--explain-method simple_gradient
```

## Citation
If you find our work useful, please cite:
```bibtex
@inproceedings{khoatn-mmm2027,
  title={Few Pixels, Fragile Faithfulness: Evaluating Visual Explanations under Sparse Perturbations},
  author={Khoa Tran, Viet-Hang Duong, Ngoc Hoang Luong},
  booktitle={International Conference on Multimedia Modeling (MM)},
  year={2026},
  publisher = {Springer}
}

@inproceedings{sparse_saliency,
  title     = {On the Instability of Saliency Maps under Sparse Perturbations},
  author    = {Khoa Tran and Thai Huy Nguyen and Quan Minh Phan and Khang Tran and Viet-Hang Duong and Tien-Dung Mai and Ngoc Hoang Luong},
  booktitle = {Proceedings of the 2026 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  year      = {2027},
  publisher = {IEEE}
}
```