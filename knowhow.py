"""
Know-how 隐性经验库
==================================================
内容：从实操/踩坑中总结的默认参数、判断门限与策略，不是书本强制规则。
来源：初始版本为工程师经验；运行后由 Lesson & Learn 持续更新。
"""

# ---------- 网格质量门限（经验值） ----------
MESH_QUALITY = {
    "min_jacobian": 0.7,     # 雅克比 > 0.7
    "max_aspect": 5.0,       # 长宽比 < 5
    "max_skew_deg": 30.0,    # 扭曲角 < 30°
    "layers_through_height": 4,  # 梁高方向至少 4 层单元
}

# ---------- 网格尺寸策略 ----------
MESH_STRATEGY = {
    # 基准网格尺寸 ≈ 梁高 / 层数（保证高向至少4层）
    "size_from_h": 4.0,
    "refine_ratio": 0.5,          # 局部加密为基准尺寸的 1/2
    "refine_zone_length_h": 1.5,  # 固定端加密区长度 = 1.5h
    "coarse_factor": 2.0,         # 粗网格调试尺寸放大倍数
}

# ---------- 加载策略 ----------
LOAD_STRATEGY = {
    # 集中力必须分布到端面多个节点（耦合），禁止单点加载造成应力奇异
    "distribute_on_end_face": True,
}

# ---------- 求解/优化策略 ----------
SOLVE_STRATEGY = {
    "first_coarse_then_fine": True,   # 先粗后细
    "max_retry": 2,                    # 单样本最大重试次数
    "penalty_scale": 1e6,              # 坏样本/约束违反惩罚量级
}

# ---------- 收敛判断 ----------
CONVERGENCE = {
    "max_iter": 35,
    "patience": 6,          # 连续 6 轮最优改善小于容差即收敛
    "tol": 1e-4,            # 相对改善容差
}


def mesh_size_for(h):
    """根据梁高给出基准网格尺寸，保证高向至少 4 层单元"""
    return h / MESH_STRATEGY["size_from_h"]


def geometric_precheck(h, t, bounds):
    """
    几何可行性预检（Agent 在建模前调用，减少无效求解）
    返回 (ok, reason)
    """
    h_lo, h_hi = bounds["h"]
    t_lo, t_hi = bounds["t"]
    if not (h_lo <= h <= h_hi):
        return False, f"h={h:.4f} 超出边界 [{h_lo},{h_hi}]"
    if not (t_lo <= t <= t_hi):
        return False, f"t={t:.4f} 超出边界 [{t_lo},{t_hi}]"
    # 经验：细长比过大网格容易畸变
    if h / t > 25.0:
        return False, f"h/t={h/t:.1f} > 25，几何过于细长，网格高风险"
    return True, "ok"
