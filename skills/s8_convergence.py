"""
S8 收敛判断 Skill
==================================================
终止条件（任一满足）：
  1) 达到最大迭代 max_iter
  2) 可行最优值连续 patience 轮相对改善 < tol
"""
from .base import BaseSkill
import knowhow


class ConvergenceSkill(BaseSkill):
    skill_name = "s8_convergence"

    def _execute(self, params):
        iteration = int(params["iteration"])
        history = params["history"]
        cfg = knowhow.CONVERGENCE
        max_iter = int(params.get("max_iter", cfg["max_iter"]))
        patience = int(params.get("patience", cfg["patience"]))
        tol = float(params.get("tol", cfg["tol"]))

        if iteration >= max_iter:
            return self.success(data={"stop": True, "reason": f"达到最大迭代 {max_iter}"})

        # 可行样本的运行最优序列
        best_so_far, curve = float("inf"), []
        for rec in history:
            if rec.get("feasible"):
                best_so_far = min(best_so_far, rec["mass"])
                curve.append(best_so_far)

        if len(curve) >= patience + 1:
            old = curve[-patience - 1]
            new = curve[-1]
            if old > 0 and (old - new) / old < tol:
                return self.success(data={"stop": True,
                                          "reason": f"连续 {patience} 轮改善 < {tol}"})
        return self.success(data={"stop": False, "reason": "继续迭代"})
