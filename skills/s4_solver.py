"""
S4 FEA 求解 Skill（自研 CST 平面应力有限元求解器）
==================================================
单元：3 节点常应变三角形 (CST)，单元刚度 ke = t*A*B^T D B
组装：scipy 稀疏矩阵；边界：消去固定自由度；求解：spsolve
输入：points, tris, t, 材料 E/nu, fixed_dofs, force
输出：节点位移向量 u、整体刚度矩阵条件信息
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from .base import BaseSkill, ERR_SOLVE
from knowledge import plane_stress_D


class FEASolveSkill(BaseSkill):
    skill_name = "s4_fea_solve"

    def _execute(self, params):
        points = params["points"]
        tris = params["tris"]
        t = float(params["t"])
        E = float(params["E"])
        nu = float(params["nu"])
        fixed_dofs = np.asarray(params["fixed_dofs"], dtype=int)
        force = np.asarray(params["force"], dtype=float)

        n = len(points)
        ndof = 2 * n
        D = plane_stress_D(E, nu)

        rows, cols, vals = [], [], []
        for tri in tris:
            p = points[tri]
            x1, y1 = p[0]
            x2, y2 = p[1]
            x3, y3 = p[2]
            beta = np.array([y2 - y3, y3 - y1, y1 - y2])
            gamma = np.array([x3 - x2, x1 - x3, x2 - x1])
            twoA = beta[0] * gamma[1] - beta[1] * gamma[0]
            if abs(twoA) < 1e-15:
                continue
            A = 0.5 * twoA
            B = np.zeros((3, 6))
            for i in range(3):
                B[0, 2 * i] = beta[i]
                B[1, 2 * i + 1] = gamma[i]
                B[2, 2 * i] = gamma[i]
                B[2, 2 * i + 1] = beta[i]
            B /= twoA
            ke = t * A * (B.T @ D @ B)

            dofs = np.array([[2 * ni, 2 * ni + 1] for ni in tri]).reshape(-1)
            for i in range(6):
                for j in range(6):
                    rows.append(dofs[i])
                    cols.append(dofs[j])
                    vals.append(ke[i, j])

        K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

        free = np.setdiff1d(np.arange(ndof), fixed_dofs)
        if len(free) == 0:
            return self.fail(ERR_SOLVE, "全部自由度被固定，检查边界条件")

        u = np.zeros(ndof)
        try:
            u_free = spsolve(K[free][:, free], force[free])
        except Exception as exc:
            return self.fail(ERR_SOLVE, f"稀疏求解失败: {exc}")
        if not np.all(np.isfinite(u_free)):
            return self.fail(ERR_SOLVE, "位移解含 NaN/Inf，求解不收敛")
        u[free] = u_free

        data = {"u": u, "K_nnz": K.nnz, "n_dof": ndof, "n_free_dof": len(free)}
        return self.success(data=data, msg=f"求解完成，自由自由度 {len(free)}")
