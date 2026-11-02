"""
S5 结果提取 Skill
==================================================
从位移解计算：单元应力分量、Mises 应力、节点平滑应力、端部挠度、质量。
Mises(平面) = sqrt(sx^2 - sx*sy + sy^2 + 3*txy^2)
"""
import numpy as np
from .base import BaseSkill
from knowledge import plane_stress_D


class ResultExtractSkill(BaseSkill):
    skill_name = "s5_result_extract"

    def _execute(self, params):
        points = params["points"]
        tris = params["tris"]
        t = float(params["t"])
        u = np.asarray(params["u"])
        L = float(params["L"])
        E, nu = float(params["E"]), float(params["nu"])
        rho = float(params["rho"])
        D = plane_stress_D(E, nu)

        n = len(points)
        stress_elem = np.zeros((len(tris), 3))   # sx, sy, txy
        mises_elem = np.zeros(len(tris))

        for k, tri in enumerate(tris):
            p = points[tri]
            (x1, y1), (x2, y2), (x3, y3) = p
            beta = np.array([y2 - y3, y3 - y1, y1 - y2])
            gamma = np.array([x3 - x2, x1 - x3, x2 - x1])
            twoA = beta[0] * gamma[1] - beta[1] * gamma[0]
            B = np.zeros((3, 6))
            for i in range(3):
                B[0, 2 * i] = beta[i]
                B[1, 2 * i + 1] = gamma[i]
                B[2, 2 * i] = gamma[i]
                B[2, 2 * i + 1] = beta[i]
            B /= twoA
            dofs = np.array([[2 * ni, 2 * ni + 1] for ni in tri]).reshape(-1)
            sig = D @ (B @ u[dofs])
            stress_elem[k] = sig
            sx, sy, txy = sig
            mises_elem[k] = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * txy ** 2)

        # 节点平滑应力（相邻单元面积加权平均，云图用）
        node_mises = np.zeros(n)
        node_count = np.zeros(n)
        for tri, m in zip(tris, mises_elem):
            for ni in tri:
                node_mises[ni] += m
                node_count[ni] += 1
        node_mises /= np.maximum(node_count, 1)

        # 端部挠度：自由端节点 y 向位移（向下为负，取绝对值最大）
        right = np.where(np.abs(points[:, 0] - L) <= 1e-9)[0]
        tip_disp = u[2 * right + 1]
        max_tip = float(np.abs(tip_disp).max())

        mass = rho * L * float(params["h"]) * t

        data = {
            "stress_elem": stress_elem,
            "mises_elem": mises_elem,
            "node_mises": node_mises,
            "max_mises": float(mises_elem.max()),
            "tip_deflection": max_tip,
            "mass": mass,
            "disp_mag": np.sqrt(u[0::2] ** 2 + u[1::2] ** 2),
        }
        return self.success(data=data,
                            msg=f"max Mises={data['max_mises']/1e6:.1f}MPa, 挠度={max_tip*1000:.3f}mm")
