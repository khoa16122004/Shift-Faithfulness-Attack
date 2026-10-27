import math

import numpy as np
import torch
from tqdm import tqdm

from operators import build_pixel_sampling_probs, generate_offspring, init_population


class Weighted_Sum_GA:
    def __init__(self, params):
        self.params = params
        self.device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.operator_strategy = params.get("operator_strategy", "uniform")
        self.saliency_temperature = params.get("saliency_temperature", 1.0)
        if params["pop_size"] < 1 or params["iterations"] < 1:
            raise ValueError("pop_size and iterations must both be >= 1")
        if not 1 <= params["eps"] <= params["all_pixels"].numel():
            raise ValueError("eps must be between 1 and the number of image pixels")
        self.init_pixel_probs = None
        if self.operator_strategy == "saliency_guided":
            self.init_pixel_probs = build_pixel_sampling_probs(
                params["fitness"].saliency_true[0],
                params["all_pixels"],
                temperature=self.saliency_temperature,
            )

    @staticmethod
    def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        order = np.argsort(x, kind="mergesort")
        sorted_vals = x[order]
        ranks = np.empty_like(sorted_vals, dtype=np.float64)
        i = 0
        while i < sorted_vals.size:
            j = i + 1
            while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) * 0.5
            i = j
        return ranks

    @classmethod
    def _spearman_rank_corr(cls, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.size != y.size or x.size < 2:
            return float("nan")
        rx = cls._rankdata_average_ties(x)
        ry = cls._rankdata_average_ties(y)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
        return float("nan") if denom < 1e-12 else float(np.dot(rx, ry) / denom)

    def _compute_history_spearman(self, candidate):
        fitness = self.params["fitness"]
        with torch.enable_grad():
            saliency, _ = fitness.explain_method(
                fitness.model,
                candidate.generate_adv_image().unsqueeze(0),
                fitness.normalize,
                fitness.y_true,
            )
        return self._spearman_rank_corr(
            fitness.saliency_true[0].detach().float().cpu().numpy(),
            saliency[0].detach().float().cpu().numpy(),
        )

    def _compute_weighted_fitness(self, objectives):
        saliency = objectives.get("saliency_loss")
        if "del_loss" not in objectives:
            if saliency is None:
                raise ValueError("Fitness must provide saliency or deletion/insertion objectives")
            return float(self.params["w_saliency"]) * saliency

        deletion = objectives["del_loss"]
        insertion = objectives["ins_loss"]
        w_del = self.params.get("w_del")
        w_ins = self.params.get("w_ins")
        w_causal = self.params.get("w_causal")
        score = torch.zeros_like(deletion)
        if saliency is not None:
            score = score + float(self.params["w_saliency"]) * saliency
        if w_del is not None or w_ins is not None:
            return score + float(w_del or 0.0) * deletion + float(w_ins or 0.0) * insertion
        causal_weight = float(self.params["w_saliency"] if w_causal is None else w_causal)
        return score + causal_weight * (deletion + insertion) * 0.5

    def _evaluate(self, population):
        images = torch.stack([candidate.generate_adv_image() for candidate in population])
        objectives, logits = self.params["fitness"].benchmark_objectives(images)
        objectives = {name: values.detach() for name, values in objectives.items()}
        scores = self._compute_weighted_fitness(objectives)
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite weighted fitness encountered")
        labels = self.params["fitness"].y_true
        preserving = logits.detach().argmax(dim=1).eq(labels)
        for idx, candidate in enumerate(population):
            candidate.pred_label = int(logits[idx].argmax().item())
            candidate.is_adversarial = not bool(preserving[idx].item())
        return objectives, scores, preserving

    @staticmethod
    def _pick_final_candidate_idx(preserving, scores):
        indices = torch.where(preserving)[0]
        if indices.numel() == 0:
            raise ValueError("No class-preserving candidate available")
        return int(indices[torch.argmin(scores[indices])].item())

    def _build_best_scores(self, candidate, objectives, score, first_success_iteration):
        best_scores = {name: float(value) for name, value in objectives.items()}
        best_scores.update(
            weighted_fitness=float(score),
            first_success_iteration=first_success_iteration,
            spearman=self._compute_history_spearman(candidate),
        )
        metric_fitness = self.params.get("metric_fitness")
        if "del_loss" not in best_scores and metric_fitness is not None:
            with torch.enable_grad():
                metrics, _ = metric_fitness.benchmark_objectives(
                    candidate.generate_adv_image().unsqueeze(0)
                )
            best_scores.update(
                del_loss=float(metrics["del_loss"][0]),
                ins_loss=float(metrics["ins_loss"][0]),
            )
        return best_scores

    def attack(self):
        fitness = self.params["fitness"]
        with torch.no_grad():
            clean_pred = fitness.model(fitness.normalize(self.params["x_tensor"])).argmax(dim=1)
        if not torch.equal(clean_pred, fitness.y_true):
            raise ValueError("Class-preserving attacks require label to match the clean prediction")

        population = init_population(
            pop_size=self.params["pop_size"],
            x_tensor=self.params["x_tensor"],
            eps=self.params["eps"],
            p_size=self.params["p_size"],
            zero_prob=self.params["zero_probability"],
            all_pixels=self.params["all_pixels"],
            pixel_probs=self.init_pixel_probs,
        )
        objectives, scores, preserving = self._evaluate(population)
        if not preserving.any():
            print("No initial candidate preserves class; using the clean image as a feasible seed.")
            population[0] = population[0].copy()
            population[0].values.zero_()
            objectives, scores, preserving = self._evaluate(population)

        best_idx = self._pick_final_candidate_idx(preserving, scores)
        best_candidate = population[best_idx].copy()
        best_score = float(scores[best_idx])
        best_objectives = {name: values[best_idx].clone() for name, values in objectives.items()}
        first_success_iteration = 0
        best_scores = self._build_best_scores(
            best_candidate, best_objectives, best_score, first_success_iteration
        )
        history = [best_scores]

        for iteration in tqdm(range(1, self.params["iterations"])):
            parent_indices = torch.randint(
                len(population), (self.params["pop_size"], 2), device=self.device
            )
            parents = [(population[int(i)], population[int(j)]) for i, j in parent_indices]
            offspring = self.generate_offpsrings(parents)
            off_objectives, off_scores, off_preserving = self._evaluate(offspring)
            pool = population + offspring
            pool_objectives = {
                name: torch.cat((values, off_objectives[name]))
                for name, values in objectives.items()
            }
            pool_scores = torch.cat((scores, off_scores))
            pool_preserving = torch.cat((preserving, off_preserving))
            candidate_idx = self._pick_final_candidate_idx(pool_preserving, pool_scores)
            if float(pool_scores[candidate_idx]) < best_score:
                best_candidate = pool[candidate_idx].copy()
                best_score = float(pool_scores[candidate_idx])
                best_objectives = {
                    name: values[candidate_idx].clone() for name, values in pool_objectives.items()
                }
                best_scores = self._build_best_scores(
                    best_candidate, best_objectives, best_score, first_success_iteration
                )
            history.append(dict(best_scores))

            selection_scores = pool_scores.masked_fill(~pool_preserving, float("inf"))
            winner_indices = self.tournament_selection(selection_scores)
            winner_indices[0] = candidate_idx  # Preserve the elite for reproduction, too.
            population = [pool[int(idx)] for idx in winner_indices]
            objectives = {name: values[winner_indices] for name, values in pool_objectives.items()}
            scores = pool_scores[winner_indices]
            preserving = pool_preserving[winner_indices]
            print(f"Iteration {iteration}: best class-preserving weighted fitness = {best_score:.4f}")

        return best_candidate.generate_adv_image(), best_candidate, best_scores, history

    def tournament_selection(self, fitnesses):
        tournaments = torch.randint(
            len(fitnesses), (self.params["pop_size"], 4), device=self.device
        )
        winners = fitnesses[tournaments].argmin(dim=1)
        return tournaments[torch.arange(self.params["pop_size"], device=self.device), winners]

    def generate_offpsrings(self, parents):
        return generate_offspring(
            parents=parents,
            pc=self.params["pc"],
            pm=self.params["pm"],
            all_pixels=self.params["all_pixels"],
            zero_prob=self.params["zero_probability"],
            pixel_probs=None,
        )
