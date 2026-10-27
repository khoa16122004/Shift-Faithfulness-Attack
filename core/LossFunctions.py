import torch
import math

class MarginSalinecy_Fitness:
    def __init__(self, model, x_tensor, normalize, y_true, explain_method):
        self.model = model
        self.y_true = y_true
        self.normalize = normalize
        self.explain_method = explain_method
        # print(x_tensor.shape)
        self.saliency_true, _ = self.explain_method(self.model, x_tensor, self.normalize,  self.y_true)
        # print("Saliency map shape: ", self.saliency_true.shape)
        # print("Diff clean: ", self.cal_saliency_loss(self.saliency_true, self.saliency_true))
        # raise
        
    def benchmark(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        margin_loss = self.cal_marginloss(logits, self.y_true)
        saliency_loss = self.cal_saliency_loss(saliency_maps, self.saliency_true)
        return margin_loss, saliency_loss, logits

    def benchmark_objectives(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        return {"saliency_loss": self.cal_saliency_loss(saliency_maps, self.saliency_true)}, logits
    
        
    def cal_marginloss(self, logits, y_true):
        with torch.no_grad():
            if y_true.numel() == 1:
                y_true = y_true.expand(logits.size(0))
            true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
            tmp = logits.clone()
            tmp.scatter_(1, y_true.unsqueeze(1), float("-inf"))
            max_other_logits = tmp.max(dim=1).values
            margin = true_logits - max_other_logits

            return margin
    
    
    def cal_saliency_loss(self, saliency_maps, saliency_true, eps=1e-12):
        with torch.no_grad():
            saliency_maps = saliency_maps.flatten(start_dim=1)
            saliency_true = saliency_true.flatten(start_dim=1)
            if saliency_true.size(0) == 1 and saliency_maps.size(0) > 1:
                saliency_true = saliency_true.expand(saliency_maps.size(0), -1)
            inter = torch.minimum(saliency_maps, saliency_true).sum(dim=1)
            union = torch.maximum(saliency_maps, saliency_true).sum(dim=1)
            soft_iou = inter / (union + eps)
            return soft_iou
    
class NegativeCrossEntropySaliency_Fitness(MarginSalinecy_Fitness):
    def benchmark(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        negative_ce_loss = self.cal_cross_entropy(logits, self.y_true)
        saliency_loss = self.cal_saliency_loss(saliency_maps, self.saliency_true)
        return negative_ce_loss, saliency_loss, logits

    def cal_cross_entropy(self, logits, y_true):
        with torch.no_grad():
            if y_true.numel() == 1:
                y_true = y_true.expand(logits.size(0))
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            negative_ce_loss = log_probs.gather(1, y_true.unsqueeze(1)).squeeze(1) + 1e-12
            return negative_ce_loss
    
    

# Increase Confidence, Decrease Saliency
class ReverseMarginSalinecy_Fitness(MarginSalinecy_Fitness):
    def __init__(self, model, x_tensor, normalize, y_true, explain_method):
        super().__init__(model, x_tensor, normalize, y_true, explain_method)
        
    def benchmark(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        margin_loss = -self.cal_marginloss(logits, self.y_true)  # Reverse the margin loss
        saliency_loss = self.cal_saliency_loss(saliency_maps, self.saliency_true)
        return margin_loss, saliency_loss, logits
    
class ReverseNegativeCrossEntropySaliency_Fitness(NegativeCrossEntropySaliency_Fitness):
    def __init__(self, model, x_tensor, normalize, y_true, explain_method):
        super().__init__(model, x_tensor, normalize, y_true, explain_method)
        
    def benchmark(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        negative_ce_loss = -self.cal_cross_entropy(logits, self.y_true)  # Reverse the negative cross-entropy loss
        saliency_loss = self.cal_saliency_loss(saliency_maps, self.saliency_true)
        return negative_ce_loss, saliency_loss, logits
    
    
# CE function to preserve original class
class CEMarrginLossSaliency_Fitness(MarginSalinecy_Fitness):
    def __init__(self, model, x_tensor, normalize, y_true, explain_method):
        super().__init__(model, x_tensor, normalize, y_true, explain_method)
            
    def benchmark(self, xadv_tensors):
        saliency_map, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        ce_margin_loss = self.cal_ce_margin_loss(logits, self.y_true)
        saliency_loss = self.cal_mse_saliency_loss(saliency_map, self.saliency_true)
        return ce_margin_loss, saliency_loss, logits

    def benchmark_objectives(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        return {"saliency_loss": self.cal_mse_saliency_loss(saliency_maps, self.saliency_true)}, logits
    
    def cal_ce_margin_loss(self, logits, y_true):
        with torch.no_grad():
            if y_true.numel() == 1:
                y_true = y_true.expand(logits.size(0))
            # Minimize CE of the original class to preserve class prediction.
            ce_loss = torch.nn.functional.cross_entropy(logits, y_true, reduction='none')
            return ce_loss

    def cal_mse_saliency_loss(self, saliency_maps, saliency_true):
        with torch.no_grad():
            saliency_maps = saliency_maps.flatten(start_dim=1)
            saliency_true = saliency_true.flatten(start_dim=1)
            if saliency_true.size(0) == 1 and saliency_maps.size(0) > 1:
                saliency_true = saliency_true.expand(saliency_maps.size(0), -1)
            mse_saliency_loss = -torch.mean((saliency_maps - saliency_true) ** 2, dim=1)
            return mse_saliency_loss



# Backward-compatible aliases.
CEMarginLossSaliency_Fitness = CEMarrginLossSaliency_Fitness
MarginLosssMSESaliency_Fitness = CEMarrginLossSaliency_Fitness


# Delection/Insertion Oriented
class MarginLossCausalFaithFull(MarginSalinecy_Fitness):
    def __init__(self, model, x_tensor, normalize, y_true, explain_method):
        super().__init__(model, x_tensor, normalize, y_true, explain_method)
        
    def benchmark(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        margin_loss = self.cal_marginloss(logits, self.y_true)
        del_loss, ins_loss = self.cal_faithfulnessloss(xadv_tensors, saliency_maps, self.y_true)
        return margin_loss, del_loss, ins_loss, logits

    def benchmark_objectives(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        del_loss, ins_loss = self.cal_faithfulnessloss(xadv_tensors, saliency_maps, self.y_true)
        return {"del_loss": del_loss, "ins_loss": ins_loss}, logits
    
    def cal_faithfulnessloss(
        self,
        x_tensors,
        saliency_maps,
        y_true,
        n=10,
        K=1.0,
        temperature=1.0,
        with_softmax=False,
        eps=1e-8,
    ):
        """
        Soft insertion/deletion objectives for minimizing faithfulness on the true class.

        Minimization direction:
        - ins_loss = mean p_true(soft-insertion): minimizing makes insertion score lower.
        - del_loss = -mean p_true(soft-deletion): minimizing makes deletion score higher.
        """

        if y_true.numel() == 1:
            y_true = y_true.expand(x_tensors.size(0))

        if saliency_maps.dim() == 4:
            if saliency_maps.size(1) == 1:
                saliency_maps = saliency_maps.squeeze(1)
            else:
                raise ValueError("Expected saliency maps as (B,H,W) or (B,1,H,W)")

        B, C, H, W = x_tensors.shape
        if saliency_maps.size(0) == 1 and B > 1:
            saliency_maps = saliency_maps.expand(B, -1, -1)

        if with_softmax:
            expl_flat = torch.log_softmax(saliency_maps.reshape(B, -1), dim=-1)
        else:
            expl_flat = saliency_maps.reshape(B, -1)
        norm_expl = expl_flat.reshape(B, H, W)

        total_pixels = H * W
        step = max(1, total_pixels // n)
        K_ = int(K * total_pixels) if isinstance(K, float) else int(K)
        if isinstance(K, float) and K == 1.0:
            maxiter = n
        else:
            maxiter = max(1, math.ceil(K_ / step))

        with torch.no_grad():
            sorted_expl, _ = torch.sort(expl_flat, dim=-1, descending=True)
            if sorted_expl.size(1) > 1:
                diff = sorted_expl[:, :-1] - sorted_expl[:, 1:]
                diff_mask = diff.abs() > 0
                diff_count = diff_mask.sum(dim=1).clamp(min=1)
                sigmoid_scale = (diff * diff_mask).sum(dim=1) / diff_count
            else:
                sigmoid_scale = torch.ones(B, device=sorted_expl.device, dtype=sorted_expl.dtype)

            steps = torch.arange(1, maxiter + 1, device=x_tensors.device, dtype=torch.long) * step
            upper_idx = torch.clamp(steps, max=total_pixels - 1)
            lower_idx = torch.clamp(upper_idx - 1, min=0)
            ts = (sorted_expl[:, lower_idx] + sorted_expl[:, upper_idx]) / 2.0

        scale = (temperature / sigmoid_scale.clamp_min(eps)).reshape(B, 1, 1, 1)
        alphas = torch.sigmoid(scale * (norm_expl.unsqueeze(1) - ts.unsqueeze(-1).unsqueeze(-1)))
        alphas = alphas.unsqueeze(2)  # (B, M, 1, H, W)

        x_expanded = x_tensors.unsqueeze(1)  # (B, 1, C, H, W)
        bg = torch.zeros((1, 1, C, 1, 1), dtype=x_tensors.dtype, device=x_tensors.device)

        x_batch_del = alphas * bg + (1 - alphas) * x_expanded
        x_batch_ins = alphas * x_expanded + (1 - alphas) * bg

        BM = B * maxiter
        labels = y_true.detach().clone().repeat_interleave(maxiter)

        x_batch_del = x_batch_del.reshape(BM, C, H, W)
        x_batch_ins = x_batch_ins.reshape(BM, C, H, W)

        with torch.no_grad():
            logits_del = self.model(self.normalize(x_batch_del))
            logits_ins = self.model(self.normalize(x_batch_ins))

        probs_del = torch.softmax(logits_del, dim=-1)[torch.arange(BM, device=x_tensors.device), labels]
        probs_ins = torch.softmax(logits_ins, dim=-1)[torch.arange(BM, device=x_tensors.device), labels]

        probs_del = probs_del.reshape(B, maxiter)
        probs_ins = probs_ins.reshape(B, maxiter)

        mean_del = probs_del.mean(dim=1)
        mean_ins = probs_ins.mean(dim=1)

        del_loss = -mean_del
        ins_loss = mean_ins

        return del_loss.detach(), ins_loss.detach()


# Backward-compatible alias.
CELossCausalFaithFull = MarginLossCausalFaithFull


class MarginLossCausalFaithFullSaliency_Fitness(MarginLossCausalFaithFull):
    """
    Combined objective variant that keeps explicit saliency loss together with
    faithfulness deletion/insertion objectives.
    """

    def benchmark(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        margin_loss = self.cal_marginloss(logits, self.y_true)
        saliency_loss = self.cal_saliency_loss(saliency_maps, self.saliency_true)
        del_loss, ins_loss = self.cal_faithfulnessloss(xadv_tensors, saliency_maps, self.y_true)
        return margin_loss, saliency_loss, del_loss, ins_loss, logits

    def benchmark_objectives(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        del_loss, ins_loss = self.cal_faithfulnessloss(xadv_tensors, saliency_maps, self.y_true)
        return {
            "saliency_loss": self.cal_saliency_loss(saliency_maps, self.saliency_true),
            "del_loss": del_loss,
            "ins_loss": ins_loss,
        }, logits


# Backward-compatible alias.
MarginLossCausalFaithFullWithSaliency_Fitness = MarginLossCausalFaithFullSaliency_Fitness



# =======================, threeobjective
            