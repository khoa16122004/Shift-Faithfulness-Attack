import argparse
import ast
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from LossFunctions import (
    CEMarrginLossSaliency_Fitness,
    MarginLossCausalFaithFull,
    MarginLossCausalFaithFullSaliency_Fitness,
    MarginSalinecy_Fitness,
    NegativeCrossEntropySaliency_Fitness,
    ReverseMarginSalinecy_Fitness,
    ReverseNegativeCrossEntropySaliency_Fitness,
)
from Solutions import Solution
from weightedSUM_GA import Weighted_Sum_GA


def load_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


class ToyModel:
    def __call__(self, images):
        codes = ((1 - images[:, 0, 0, 0]) * 10).round().long()
        preserving = ~torch.isin(codes, torch.tensor([2, 6]))
        return torch.stack((preserving.float(), (~preserving).float()), dim=1)


class ToyFitness:
    def __init__(self):
        self.model = ToyModel()
        self.normalize = lambda images: images
        self.y_true = torch.tensor([0])
        self.saliency_true = torch.arange(4).reshape(1, 2, 2).float()
        self.explain_method = self.explain

    def explain(self, model, images, normalize, labels):
        return self.saliency_true.expand(images.shape[0], -1, -1), model(normalize(images))

    def benchmark_objectives(self, images):
        codes = ((1 - images[:, 0, 0, 0]) * 10).round().long()
        losses = torch.tensor([1.0, 0.6, -9.0, 0.4, 0.8, 0.2, -10.0, 0.9])
        return {"saliency_loss": losses[codes]}, self.model(images)

    def benchmark(self, images):
        raise AssertionError("The margin-based benchmark must not be used")


class WeightedSumTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.ones(1, 3, 2, 2)
        self.params = {
            "fitness": ToyFitness(),
            "x_tensor": self.image,
            "pop_size": 2,
            "iterations": 3,
            "eps": 1,
            "p_size": 1.0,
            "zero_probability": 0.3,
            "all_pixels": torch.arange(4),
            "w_saliency": 1.0,
            "pc": 0.4,
            "pm": 0.1,
            "device": "cpu",
        }

    def candidate(self, code):
        return Solution(
            torch.tensor([0]),
            torch.tensor([[code / 10, 0.0, 0.0]]),
            self.image.clone(),
            1.0,
        )

    def test_global_best_preserves_class_and_ignores_better_invalid_scores(self):
        attacker = Weighted_Sum_GA(self.params)
        with patch("weightedSUM_GA.init_population", return_value=[self.candidate(1), self.candidate(2)]), patch.object(
            attacker, "generate_offpsrings",
            side_effect=[
                [self.candidate(3), self.candidate(6)],
                [self.candidate(4), self.candidate(7)],
            ],
        ), patch.object(attacker, "tournament_selection", return_value=torch.tensor([1, 1])):
            image, candidate, scores, history = attacker.attack()
        self.assertFalse(candidate.is_adversarial)
        self.assertAlmostEqual(float(image[0, 0, 0]), 0.7)
        self.assertAlmostEqual(scores["weighted_fitness"], 0.4)
        self.assertEqual(len(history), 3)
        self.assertEqual(
            [round(row["weighted_fitness"], 2) for row in history], [0.6, 0.4, 0.4]
        )
        self.assertNotIn("margin_loss", scores)

    def test_initialization_and_single_iteration_choose_valid_minimum(self):
        self.params["iterations"] = 1
        with patch("weightedSUM_GA.init_population", return_value=[self.candidate(2), self.candidate(1)]):
            _, candidate, scores, history = Weighted_Sum_GA(self.params).attack()
        self.assertFalse(candidate.is_adversarial)
        self.assertAlmostEqual(scores["weighted_fitness"], 0.6)
        self.assertEqual(len(history), 1)

    def test_no_feasible_initial_candidate_uses_clean_seed_then_improves(self):
        self.params["iterations"] = 2
        attacker = Weighted_Sum_GA(self.params)
        with patch("weightedSUM_GA.init_population", return_value=[self.candidate(2), self.candidate(6)]), patch.object(
            attacker, "generate_offpsrings", return_value=[self.candidate(5), self.candidate(6)]
        ):
            _, candidate, scores, history = attacker.attack()
        self.assertEqual(history[0]["weighted_fitness"], 1.0)
        self.assertAlmostEqual(scores["weighted_fitness"], 0.2)
        self.assertFalse(candidate.is_adversarial)

    def test_all_invalid_single_iteration_returns_clean_image(self):
        self.params["iterations"] = 1
        with patch("weightedSUM_GA.init_population", return_value=[self.candidate(2), self.candidate(6)]):
            image, candidate, _, _ = Weighted_Sum_GA(self.params).attack()
        self.assertTrue(torch.equal(image, self.image[0]))
        self.assertEqual(int(candidate.l0_distance(image)), 0)

    def test_wrong_label_is_an_explicit_error(self):
        self.params["fitness"].y_true = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, "label"):
            Weighted_Sum_GA(self.params).attack()

    def test_nonfinite_fitness_is_an_explicit_error(self):
        self.params["w_saliency"] = math.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            Weighted_Sum_GA(self.params)._evaluate([self.candidate(1)])

    def test_weighted_sum_formulas_and_explicit_weights_take_precedence(self):
        self.params.update(w_saliency=2.0, w_causal=3.0)
        attacker = Weighted_Sum_GA(self.params)
        objectives = {
            "saliency_loss": torch.tensor([0.25]),
            "del_loss": torch.tensor([-0.8]),
            "ins_loss": torch.tensor([0.2]),
        }
        self.assertAlmostEqual(float(attacker._compute_weighted_fitness(objectives)), -0.4, places=6)
        self.params.update(w_del=0.25, w_ins=0.75)
        self.assertAlmostEqual(float(attacker._compute_weighted_fitness(objectives)), 0.45, places=6)
        del objectives["saliency_loss"]
        self.assertAlmostEqual(float(attacker._compute_weighted_fitness(objectives)), -0.05, places=6)
        self.params.update(w_del=None, w_ins=None, w_causal=None)
        self.assertAlmostEqual(float(attacker._compute_weighted_fitness(objectives)), -0.6, places=6)

    def test_tournament_preserves_population_size_for_odd_and_small_sizes(self):
        for size in (1, 2, 3, 5, 50):
            with self.subTest(size=size):
                self.params["pop_size"] = size
                result = Weighted_Sum_GA(self.params).tournament_selection(torch.arange(2 * size).float())
                self.assertEqual(result.shape, (size,))
                self.assertTrue(((result >= 0) & (result < 2 * size)).all())

    def test_all_fitness_variants_skip_classification_loss(self):
        fitness_types = (
            MarginSalinecy_Fitness,
            NegativeCrossEntropySaliency_Fitness,
            ReverseMarginSalinecy_Fitness,
            ReverseNegativeCrossEntropySaliency_Fitness,
            CEMarrginLossSaliency_Fitness,
            MarginLossCausalFaithFull,
            MarginLossCausalFaithFullSaliency_Fitness,
        )
        toy = self.params["fitness"]
        for fitness_type in fitness_types:
            with self.subTest(fitness=fitness_type.__name__):
                fitness = fitness_type(
                    model=toy.model, x_tensor=self.image, normalize=toy.normalize,
                    y_true=toy.y_true, explain_method=toy.explain,
                )
                with patch.object(fitness, "cal_marginloss", side_effect=AssertionError("margin evaluated")):
                    if isinstance(fitness, MarginLossCausalFaithFull):
                        with patch.object(
                            fitness, "cal_faithfulnessloss",
                            return_value=(torch.tensor([-0.8]), torch.tensor([0.2])),
                        ):
                            objectives, logits = fitness.benchmark_objectives(self.image)
                    else:
                        objectives, logits = fitness.benchmark_objectives(self.image)
                self.assertNotIn("margin_loss", objectives)
                self.assertEqual(logits.argmax(dim=1).item(), 0)
                if isinstance(fitness, CEMarrginLossSaliency_Fitness):
                    self.assertEqual(float(objectives["saliency_loss"][0]), 0.0)

    def test_metric_logging_does_not_change_optimized_weighted_sum(self):
        self.params["iterations"] = 1
        metric = MarginLossCausalFaithFull(
            model=self.params["fitness"].model,
            x_tensor=self.image,
            normalize=self.params["fitness"].normalize,
            y_true=self.params["fitness"].y_true,
            explain_method=self.params["fitness"].explain,
        )
        self.params["metric_fitness"] = metric
        with patch("weightedSUM_GA.init_population", return_value=[self.candidate(1), self.candidate(2)]), patch.object(
            metric, "cal_faithfulnessloss",
            return_value=(torch.tensor([-0.8]), torch.tensor([0.2])),
        ):
            _, _, scores, _ = Weighted_Sum_GA(self.params).attack()
        self.assertAlmostEqual(scores["weighted_fitness"], 0.6)
        self.assertAlmostEqual(scores["del_loss"], -0.8)
        self.assertAlmostEqual(scores["ins_loss"], 0.2)


class RunnerContractTests(unittest.TestCase):
    def test_runners_reject_nsgaii(self):
        for filename, arguments in (
            ("run_sample.py", ["--image", "unused.png"]),
            ("run_batch.py", []),
            ("run_batch_causal.py", []),
        ):
            with self.subTest(filename=filename):
                namespace = load_functions(filename, {"parse_args"}, {
                    "argparse": argparse,
                    "DEFAULT_IMAGENET_VAL_ROOT": "unused",
                    "DEFAULT_REMOTE_VAL_ROOT": "unused",
                })
                with patch.object(sys, "argv", [filename, *arguments]):
                    args = namespace["parse_args"]()
                self.assertEqual(args.algorithm, "weighted_sum_ga")
                with patch.object(sys, "argv", [filename, *arguments, "--algorithm", "nsgaii"]):
                    with self.assertRaises(SystemExit):
                        namespace["parse_args"]()

    def test_history_file_has_named_margin_free_columns(self):
        namespace = load_functions(
            "run_batch.py", {"save_history_scores_txt", "_to_float"}, {"torch": torch}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.txt"
            namespace["save_history_scores_txt"](
                [{"weighted_fitness": -0.3, "del_loss": -0.8, "ins_loss": 0.2}], path
            )
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["# weighted_fitness del_loss ins_loss", "-0.3 -0.8 0.2"])

    def test_causal_only_charts_do_not_require_saliency_or_margin(self):
        calls = []
        namespace = load_functions(
            "util.py", {"save_attack_two_score_charts"},
            {"_to_float": float, "_save_single_score_chart": lambda **kwargs: calls.append(kwargs)},
        )
        namespace["save_attack_two_score_charts"](
            [{"weighted_fitness": -0.3, "ins_loss": 0.2}], "fitness.png", "insertion.png"
        )
        self.assertEqual([call["label"] for call in calls], ["weighted_fitness", "ins_loss"])


if __name__ == "__main__":
    unittest.main()
