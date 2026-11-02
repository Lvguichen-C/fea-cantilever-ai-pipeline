"""
S6 约束评估 Skill
==================================================
判断应力/挠度是否满足门限，计算相对违反量与带惩罚的目标值。
惩罚目标 = mass + scale * (应力违反 + 挠度违反)，违反量为相对量，无量纲。
"""
from .base import BaseSkill
import knowhow


class ConstraintEvaluateSkill(BaseSkill):
    skill_name = "s6_constraint_evaluate"

    def _execute(self, params):
        mass = float(params["mass"])
        max_mises = float(params["max_mises"])
        tip = float(params["tip_deflection"])
        stress_limit = float(params["stress_limit"])
        disp_limit = float(params["disp_limit"])

        v_stress = max(0.0, max_mises / stress_limit - 1.0)
        v_disp = max(0.0, tip / disp_limit - 1.0)
        feasible = v_stress <= 0.0 and v_disp <= 0.0

        scale = knowhow.SOLVE_STRATEGY["penalty_scale"]
        penalized_obj = mass + scale * (v_stress + v_disp)

        data = {
            "feasible": feasible,
            "violation_stress": v_stress,
            "violation_disp": v_disp,
            "objective": mass,                 # 真实目标
            "penalized_objective": penalized_obj,
        }
        msg = "可行设计" if feasible else \
            f"违反约束(应力相对违反 {v_stress*100:.1f} / 挠度相对违反 {v_disp*100:.1f})"
        return self.success(data=data, msg=msg)
