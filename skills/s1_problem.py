"""
S1 需求解析 Skill
==================================================
把问题定义（可由自然语言经 LLM 转写）固化为结构化优化问题 JSON。
本教学版直接从 knowledge 读取，保证确定性；LLM 仅需替换 build_problem 的来源。
"""
from .base import BaseSkill
import knowledge


class ProblemDefineSkill(BaseSkill):
    skill_name = "s1_problem_define"

    def _execute(self, params):
        prob = knowledge.PROBLEM
        problem = {
            "L": prob["L"],
            "F": prob["F"],
            "variables": ["h", "t"],
            "bounds": prob["bounds"],
            "objective": "min_mass",
            "constraints": {
                "max_mises_stress": prob["max_stress"],
                "max_tip_deflection": knowledge.deflection_limit(prob),
            },
            "material": knowledge.STEEL["name"],
        }
        return self.success(data={"problem": problem}, msg="优化问题已结构化")
