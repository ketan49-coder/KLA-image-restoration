"""
composite_metric.py
Prevents the "40 dB PSNR but 0.12 SSIM" failure mode by scoring checkpoints
on a BALANCED combination of PSNR, SSIM, and LPIPS — not any single metric alone.
"""

class CompositeScorer:
    """
    Tracks a normalized, weighted combination of PSNR, SSIM, and LPIPS
    across training, and decides when a new checkpoint is a genuine
    all-round improvement (not just a win on one metric at another's expense).
    """

    def __init__(self, psnr_ceiling=35.0, w_psnr=0.4, w_ssim=0.35, w_lpips=0.25):
        assert abs((w_psnr + w_ssim + w_lpips) - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.psnr_ceiling = psnr_ceiling
        self.w_psnr = w_psnr
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.best_score = -float("inf")
        self.best_metrics = None

    def compute_score(self, psnr, ssim, lpips):
        psnr_norm = psnr / self.psnr_ceiling
        ssim_norm = ssim                    # already 0-1, higher is better
        lpips_norm = 1.0 - lpips             # invert: higher is better

        score = (
            self.w_psnr * psnr_norm +
            self.w_ssim * ssim_norm +
            self.w_lpips * lpips_norm
        )
        return score

    def is_new_best(self, psnr, ssim, lpips):
        score = self.compute_score(psnr, ssim, lpips)
        is_best = score > self.best_score

        breakdown = {
            "score": score,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "psnr_contribution": self.w_psnr * (psnr / self.psnr_ceiling),
            "ssim_contribution": self.w_ssim * ssim,
            "lpips_contribution": self.w_lpips * (1.0 - lpips),
        }

        if is_best:
            self.best_score = score
            self.best_metrics = breakdown

        return is_best, score, breakdown

    def state_dict(self):
        return {"best_score": self.best_score, "best_metrics": self.best_metrics}

    def load_state_dict(self, state):
        self.best_score = state["best_score"]
        self.best_metrics = state["best_metrics"]
