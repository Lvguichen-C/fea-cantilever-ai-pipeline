"""
S7 优化采样 Skill（Optuna TPE 黑盒优化）
==================================================
有状态 Skill：内部持有 study。
  suggest()：TPE 给出下一组 h,t
  report()：把带惩罚目标回传给 study（坏样本给大惩罚，驱动采样避开）
封装的 Know-how：变量归一化由 Optuna 内部处理；失败样本惩罚而非丢弃。
"""
import optuna
from .base import BaseSkill

# 静默 Optuna 日志
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptimizationSkill(BaseSkill):
    skill_name = "s7_optimization"

    def __init__(self, bounds):
        self.bounds = bounds
        self.study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
        self._trial = None

    def suggest(self):
        self._trial = self.study.ask()
        h_lo, h_hi = self.bounds["h"]
        t_lo, t_hi = self.bounds["t"]
        h = self._trial.suggest_float("h", h_lo, h_hi)
        t = self._trial.suggest_float("t", t_lo, t_hi)
        return {"h": h, "t": t}

    def report(self, penalized_objective, feasible):
        # 必须在 tell 之前写 user_attr，tell 后 trial 即结束不可更新
        self._trial.set_user_attr("feasible", bool(feasible))
        self.study.tell(self._trial, penalized_objective,
                        state=optuna.trial.TrialState.COMPLETE)

    def report_failure(self, bad_value):
        """网格/求解失败样本：给大惩罚值"""
        self._trial.set_user_attr("feasible", False)
        self.study.tell(self._trial, bad_value,
                        state=optuna.trial.TrialState.COMPLETE)

    def best(self):
        try:
            return self.study.best_trial
        except ValueError:
            return None

    def _execute(self, params):
        # 保留统一接口；实际调度通过 suggest/report 方法
        action = params.get("action", "suggest")
        if action == "suggest":
            return self.success(data=self.suggest())
        return self.success(data={})
