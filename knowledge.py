"""
Knowledge 显性知识库
==================================================
内容：可直接文字传递的事实、材料参数、解析公式、规范门限。
特点：只放「手册里查得到」的硬知识，不放假想经验值。
"""
import numpy as np

# ---------- 材料参数（结构钢，教学示例值） ----------
STEEL = {
    "name": "Structural Steel",
    "E": 206e9,          # 弹性模量 Pa
    "nu": 0.3,           # 泊松比
    "rho": 7850.0,       # 密度 kg/m^3
    "yield_strength": 235e6,
    "allowable_stress": 200e6,   # 许用应力 [sigma]=200MPa
}

# ---------- 几何与载荷（问题常量） ----------
PROBLEM = {
    "L": 1.0,            # 梁长 m
    "F": 1000.0,         # 自由端总集中力 N（向下）
    # 设计变量边界
    "bounds": {
        "h": (0.02, 0.12),    # 梁高 m
        "t": (0.004, 0.02),   # 板厚 m（平面应力厚度）
    },
    # 规范约束
    "max_stress": 200e6,     # Pa
    "max_deflection_ratio": 250.0,  # 挠度 <= L/250
}


def analytical_solution(h, t, mat=STEEL, prob=PROBLEM):
    """
    材料力学 Euler-Bernoulli 梁解析解（用于验证 FEA 结果）
    矩形截面：I = t*h^3/12
    返回：端部挠度、固定端最大弯曲应力、质量
    """
    L, F = prob["L"], prob["F"]
    E = mat["E"]
    I = t * h ** 3 / 12.0
    delta = F * L ** 3 / (3.0 * E * I)                 # 端部挠度
    sigma = F * L * (h / 2.0) / I                      # 固定端最大弯曲应力
    mass = mat["rho"] * L * h * t                      # 梁质量
    return {"delta": delta, "stress": sigma, "mass": mass, "I": I}


def deflection_limit(prob=PROBLEM):
    """规范挠度限值 L/250"""
    return prob["L"] / prob["max_deflection_ratio"]


def plane_stress_D(E, nu):
    """平面应力本构矩阵 D (3x3)"""
    c = E / (1.0 - nu ** 2)
    return c * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])
