"""
Agent —— FEA 结构优化调度智能体（包工头）
==================================================
职责：任务拆解、状态管理、Skill 编排、异常容错、迭代控制、历史持久化。
不做任何数值计算；所有计算通过 Skill 完成。
LLM 可在本文件外层替换 S1 的输入来源与报错诊断，不影响本闭环。
"""
import os
import json
import pandas as pd

import knowledge
import knowhow
from skills.s1_problem import ProblemDefineSkill
from skills.s2_mesh import MeshingSkill
from skills.s3_bc import BoundaryLoadSkill
from skills.s4_solver import FEASolveSkill
from skills.s5_extract import ResultExtractSkill
from skills.s6_evaluate import ConstraintEvaluateSkill
from skills.s7_optimizer import OptimizationSkill
from skills.s8_convergence import ConvergenceSkill
from skills.base import ERR_MESH, ERR_SOLVE


class FEAAgent:
    def __init__(self, output_dir="outputs", max_iter=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.ctx = {
            "iteration": 0,
            "history": [],
            "best_feasible": None,
            "best_data": None,
        }
        self.max_iter = max_iter or knowhow.CONVERGENCE["max_iter"]
        self._instantiate_skills()

    def _instantiate_skills(self):
        self.s1 = ProblemDefineSkill()
        self.s2 = MeshingSkill()
        self.s3 = BoundaryLoadSkill()
        self.s4 = FEASolveSkill()
        self.s5 = ResultExtractSkill()
        self.s6 = ConstraintEvaluateSkill()
        self.s8 = ConvergenceSkill()

    def run(self):
        # ---- S1：问题定义 ----
        r1 = self.s1.run({})
        problem = r1["data"]["problem"]
        bounds = problem["bounds"]
        self.s7 = OptimizationSkill(bounds)
        mat = knowledge.STEEL
        L, F = problem["L"], problem["F"]
        stress_limit = problem["constraints"]["max_mises_stress"]
        disp_limit = problem["constraints"]["max_tip_deflection"]

        print(f"[Agent] 任务启动：变量 {problem['variables']}，最大迭代 {self.max_iter}")

        while self.ctx["iteration"] < self.max_iter:
            it = self.ctx["iteration"]

            # ---- S7：采样新设计 ----
            design = self.s7.suggest()
            h, t = design["h"], design["t"]

            # ---- 几何可行性预检（Lesson&Learn 沉淀的策略） ----
            ok, reason = knowhow.geometric_precheck(h, t, bounds)
            if not ok:
                self.s7.report_failure(100.0)
                self._record(it, h, t, None, feasible=False, status="PRECHECK_FAIL", note=reason)
                self._next_step(it)
                continue

            # ---- S2：网格 ----
            r2 = self.s2.run({"h": h, "t": t, "L": L,
                              "mesh_size": knowhow.mesh_size_for(h)})
            if r2["status"] == ERR_MESH:
                self.s7.report_failure(100.0)
                self._record(it, h, t, None, feasible=False, status="MESH_FAIL", note=r2["msg"])
                self._next_step(it)
                continue

            mesh = r2["data"]

            # ---- S3：边界载荷 ----
            r3 = self.s3.run({"points": mesh["points"], "L": L, "F": F})

            # ---- S4：求解 ----
            r4 = self.s4.run({
                "points": mesh["points"], "tris": mesh["tris"], "t": t,
                "E": mat["E"], "nu": mat["nu"],
                "fixed_dofs": r3["data"]["fixed_dofs"],
                "force": r3["data"]["force"],
            })
            if r4["status"] == ERR_SOLVE:
                self.s7.report_failure(100.0)
                self._record(it, h, t, None, feasible=False, status="SOLVE_FAIL", note=r4["msg"])
                self._next_step(it)
                continue

            # ---- S5：结果提取 ----
            r5 = self.s5.run({
                "points": mesh["points"], "tris": mesh["tris"], "t": t,
                "u": r4["data"]["u"], "L": L, "E": mat["E"], "nu": mat["nu"],
                "rho": mat["rho"], "h": h,
            })
            res = r5["data"]

            # ---- S6：约束评估 ----
            r6 = self.s6.run({
                "mass": res["mass"], "max_mises": res["max_mises"],
                "tip_deflection": res["tip_deflection"],
                "stress_limit": stress_limit, "disp_limit": disp_limit,
            })
            ev = r6["data"]
            self.s7.report(ev["penalized_objective"], ev["feasible"])

            self._record(it, h, t, res, feasible=ev["feasible"],
                         status="FEASIBLE" if ev["feasible"] else "INFEASIBLE",
                         note=r6["msg"], mesh=mesh, u=r4["data"]["u"])

            # 维护可行最优
            if ev["feasible"] and (self.ctx["best_feasible"] is None
                                    or res["mass"] < self.ctx["best_feasible"]):
                self.ctx["best_feasible"] = res["mass"]
                self.ctx["best_data"] = {
                    "h": h, "t": t, "points": mesh["points"], "tris": mesh["tris"],
                    "u": r4["data"]["u"], "result": res,
                }

            # ---- S8：收敛判断 ----
            r8 = self.s8.run({
                "iteration": self.ctx["iteration"] + 1,
                "history": self.ctx["history"],
                "max_iter": self.max_iter,
            })
            print(f"[Agent] iter={it+1:02d} h={h:.4f} t={t:.4f} "
                  f"mass={res['mass']:.3f}kg stress={res['max_mises']/1e6:.1f}MPa "
                  f"disp={res['tip_deflection']*1000:.2f}mm {r6['msg']}")
            if r8["data"]["stop"]:
                print(f"[Agent] 终止：{r8['data']['reason']}")
                break
            self.ctx["iteration"] += 1

        self._persist()
        return self._summary()

    def _record(self, it, h, t, res, feasible, status, note, mesh=None, u=None):
        rec = {
            "iteration": it, "h": h, "t": t,
            "mass": res["mass"] if res else None,
            "max_mises_MPa": res["max_mises"] / 1e6 if res else None,
            "tip_deflection_mm": res["tip_deflection"] * 1000 if res else None,
            "feasible": feasible, "status": status, "note": note,
        }
        self.ctx["history"].append(rec)

    def _next_step(self, it):
        print(f"[Agent] iter={it+1:02d} 坏样本已惩罚并跳过")
        self.ctx["iteration"] += 1

    def _persist(self):
        df = pd.DataFrame(self.ctx["history"])
        csv_path = os.path.join(self.output_dir, "iteration_history.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if self.ctx["best_data"]:
            bd = {k: v for k, v in self.ctx["best_data"].items()
                  if k in ("h", "t")}
            with open(os.path.join(self.output_dir, "best_design.json"), "w") as f:
                json.dump(bd, f, ensure_ascii=False, indent=2)
        return csv_path

    def _summary(self):
        n_feasible = sum(1 for r in self.ctx["history"] if r["feasible"])
        return {
            "n_iter": len(self.ctx["history"]),
            "n_feasible": n_feasible,
            "best_mass": self.ctx["best_feasible"],
            "best_design": self.ctx["best_data"],
            "history": self.ctx["history"],
        }
