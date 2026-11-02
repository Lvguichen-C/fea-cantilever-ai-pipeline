"""
主入口：运行完整 AI 管线
==================================================
1) 网格收敛研究（h-refinement，对照解析解）
2) Agent 驱动的尺寸优化闭环（Optuna TPE + 真实 FEA）
3) 输出应力/位移云图、优化收敛曲线、网格收敛曲线、CSV
运行：python3 run_pipeline.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import knowledge
import knowhow
from agent import FEAAgent
from skills.s2_mesh import MeshingSkill
from skills.s3_bc import BoundaryLoadSkill
from skills.s4_solver import FEASolveSkill
from skills.s5_extract import ResultExtractSkill
import plotting

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def mesh_convergence_study(h=0.05, t=0.01):
    """固定设计，逐级加密网格，采集收敛数据"""
    mat = knowledge.STEEL
    ana = knowledge.analytical_solution(h, t)
    rows = []
    for factor in [1.0, 0.6, 0.4, 0.25]:
        size = knowhow.mesh_size_for(h) * factor
        r2 = MeshingSkill().run({"h": h, "t": t, "L": 1.0,
                                 "mesh_size": size, "refine": False})
        mesh = r2["data"]
        r3 = BoundaryLoadSkill().run({"points": mesh["points"], "L": 1.0, "F": 1000.0})
        r4 = FEASolveSkill().run({
            "points": mesh["points"], "tris": mesh["tris"], "t": t,
            "E": mat["E"], "nu": mat["nu"],
            "fixed_dofs": r3["data"]["fixed_dofs"], "force": r3["data"]["force"]})
        r5 = ResultExtractSkill().run({
            "points": mesh["points"], "tris": mesh["tris"], "t": t,
            "u": r4["data"]["u"], "L": 1.0, "E": mat["E"], "nu": mat["nu"],
            "rho": mat["rho"], "h": h})
        d = r5["data"]
        rows.append({"mesh_size": size, "n_nodes": mesh["n_nodes"],
                     "tip_deflection_mm": d["tip_deflection"] * 1000,
                     "max_mises_MPa": d["max_mises"] / 1e6})
        print(f"[MeshStudy] size={size:.4f} 节点={mesh['n_nodes']:4d} "
              f"挠度={d['tip_deflection']*1000:.4f}mm 应力={d['max_mises']/1e6:.2f}MPa")
    plotting.plot_mesh_convergence(
        rows, ana, os.path.join(OUT, "网格收敛曲线.png"))
    return rows, ana


def main():
    print("=" * 66)
    print("阶段 1：网格收敛性研究（h-refinement）")
    print("=" * 66)
    mesh_convergence_study()

    print("\n" + "=" * 66)
    print("阶段 2：Agent 尺寸优化闭环")
    print("=" * 66)
    agent = FEAAgent(output_dir=OUT)
    summary = agent.run()

    if summary["best_design"] is None:
        print("未找到可行设计，请放宽约束后重试")
        return

    best = summary["best_design"]
    print("\n" + "=" * 66)
    print("阶段 3：绘制云图与收敛图")
    print("=" * 66)
    plotting.plot_stress_contour(best, os.path.join(OUT, "应力云图.png"))
    plotting.plot_disp_contour(best, os.path.join(OUT, "位移云图.png"))
    plotting.plot_optimization_convergence(
        summary["history"], os.path.join(OUT, "优化收敛曲线.png"))

    # 最优设计与解析解对照
    ana = knowledge.analytical_solution(best["h"], best["t"])
    print("\n========== 最终结果 ==========")
    print(f"最优 h = {best['h']*1000:.2f} mm, t = {best['t']*1000:.2f} mm")
    print(f"质量   = {summary['best_mass']:.4f} kg")
    print(f"应力   = {best['result']['max_mises']/1e6:.2f} MPa (限值 200)")
    print(f"挠度   = {best['result']['tip_deflection']*1000:.3f} mm (限值 4)")
    print(f"同尺寸解析解: 挠度 {ana['delta']*1000:.3f}mm, 应力 {ana['stress']/1e6:.1f}MPa")
    print(f"总样本 {summary['n_iter']}，可行样本 {summary['n_feasible']}")
    print(f"产物目录: {OUT}")


if __name__ == "__main__":
    main()
