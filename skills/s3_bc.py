"""
S3 边界条件与载荷 Skill
==================================================
封装的 Know-how：集中力分布到端面全部节点（耦合加载），禁止单点应力奇异。
输入：points, L, F
输出：固定自由度列表、整体外力向量
"""
import numpy as np
from .base import BaseSkill


class BoundaryLoadSkill(BaseSkill):
    skill_name = "s3_boundary_load"

    def _execute(self, params):
        points = params["points"]
        L = float(params["L"])
        F = float(params["F"])
        tol = 1e-9
        n = len(points)

        # 固定端：x≈0 的节点，ux=uy=0
        left = np.where(np.abs(points[:, 0]) <= tol)[0]
        fixed_dofs = np.sort(np.concatenate([2 * left, 2 * left + 1]))

        # 自由端：x≈L 的节点，总力 F 均分到各节点（y 向，向下为负）
        right = np.where(np.abs(points[:, 0] - L) <= tol)[0]
        if len(right) == 0:
            raise ValueError("未找到自由端节点")
        force = np.zeros(2 * n)
        force[2 * right + 1] = -F / len(right)

        data = {
            "fixed_dofs": fixed_dofs,
            "force": force,
            "n_fixed_nodes": len(left),
            "n_loaded_nodes": len(right),
        }
        return self.success(data=data, msg=f"固定 {len(left)} 节点，载荷耦合到 {len(right)} 个端面节点")
