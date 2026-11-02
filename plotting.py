"""
绘图模块：应力云图、位移云图、优化收敛曲线、网格收敛曲线
==================================================
全部基于 matplotlib 确定性绘制，中文字体 Noto Sans CJK SC。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _tri(points, tris):
    return Triangulation(points[:, 0], points[:, 1], tris)


def plot_stress_contour(best, out_path):
    """最优设计 Mises 应力云图（含变形放大）"""
    points, tris = best["points"], best["tris"]
    res, u = best["result"], best["u"]
    tri = _tri(points, tris)

    # 变形放大系数（仅用于可视化）
    scale = 0.12 / np.abs(u).max()
    deformed = points.copy()
    deformed[:, 0] += u[0::2] * scale
    deformed[:, 1] += u[1::2] * scale

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    tpc = ax.tripcolor(deformed[:, 0], deformed[:, 1], tris,
                       facecolors=res["mises_elem"] / 1e6,
                       cmap="jet", shading="flat",
                       vmin=0, vmax=max(200, res["mises_elem"].max() / 1e6))
    cb = fig.colorbar(tpc, ax=ax, pad=0.02)
    cb.set_label("Mises 应力 (MPa)", fontsize=11)
    ax.triplot(tri, color="k", lw=0.35, alpha=0.55)
    ax.set_aspect("equal")
    ax.set_title(f"最优设计 Mises 应力云图（变形已放大）  "
                 f"h={best['h']*1000:.1f}mm, t={best['t']*1000:.1f}mm, "
                 f"max={res['max_mises']/1e6:.1f}MPa", fontsize=12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.annotate("固定端", xy=(0, best["h"] / 2), xytext=(0.06, best["h"] * 0.85),
                fontsize=10, color="#333",
                arrowprops=dict(arrowstyle="->", color="#333"))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_disp_contour(best, out_path):
    """最优设计位移幅值云图"""
    points, tris = best["points"], best["tris"]
    res = best["result"]
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    tpc = ax.tripcolor(_tri(points, tris), res["disp_mag"] * 1000,
                       cmap="viridis", shading="flat")
    cb = fig.colorbar(tpc, ax=ax, pad=0.02)
    cb.set_label("位移幅值 (mm)", fontsize=11)
    ax.set_aspect("equal")
    ax.set_title(f"最优设计位移云图  端部挠度={res['tip_deflection']*1000:.3f}mm",
                 fontsize=12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_optimization_convergence(history, out_path):
    """优化迭代收敛曲线：目标散点 + 运行最优 + 应力/挠度子图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), dpi=150,
                                   sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    it = np.array([r["iteration"] for r in history])
    mass = np.array([r["mass"] if r["mass"] is not None else np.nan
                     for r in history])
    feasible = np.array([r["feasible"] for r in history])
    failed = np.array([r["status"] in ("MESH_FAIL", "SOLVE_FAIL", "PRECHECK_FAIL")
                       for r in history])

    # 散点：可行/不可行/失败
    ax1.scatter(it[feasible], mass[feasible], c="#2e8b57", s=42,
                label="可行样本", zorder=3, edgecolors="white", lw=0.5)
    ax1.scatter(it[~feasible & ~failed], mass[~feasible & ~failed], c="#e08a2d",
                s=42, marker="s", label="约束违反样本", zorder=3,
                edgecolors="white", lw=0.5)
    if failed.any():
        ax1.scatter(it[failed], np.full(failed.sum(), mass[~np.isnan(mass)].max() * 1.08),
                    c="#c0392b", s=55, marker="x", label="失败/惩罚样本", zorder=4)

    # 运行可行最优
    best_curve, cur = [], np.inf
    for r in history:
        if r["feasible"]:
            cur = min(cur, r["mass"])
        best_curve.append(cur if np.isfinite(cur) else np.nan)
    ax1.plot(it, best_curve, color="#1f4e79", lw=2, label="运行最优（可行）", zorder=2)

    ax1.set_ylabel("质量 (kg)", fontsize=11)
    ax1.set_title("FEA 结构优化迭代收敛曲线", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3); ax1.legend(loc="upper right", fontsize=10)

    # 子图：应力与挠度
    stress = np.array([r["max_mises_MPa"] if r["max_mises_MPa"] is not None
                       else np.nan for r in history])
    disp = np.array([r["tip_deflection_mm"] if r["tip_deflection_mm"] is not None
                     else np.nan for r in history])
    ax2.plot(it, stress, "-o", color="#c0392b", ms=3.5, lw=1.2, label="最大 Mises (MPa)")
    ax2.axhline(200, color="#c0392b", ls="--", lw=1, alpha=0.7)
    ax2.set_ylabel("应力 (MPa)", color="#c0392b", fontsize=10)
    ax2b = ax2.twinx()
    ax2b.plot(it, disp, "-^", color="#2c5f8a", ms=3.5, lw=1.2,
              label="端部挠度 (mm)")
    ax2b.axhline(4, color="#2c5f8a", ls="--", lw=1, alpha=0.7)
    ax2b.set_ylabel("挠度 (mm)", color="#2c5f8a", fontsize=10)
    ax2.set_xlabel("迭代轮次", fontsize=11)
    ax2.grid(alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mesh_convergence(conv_rows, analytical, out_path):
    """网格收敛曲线：挠度/应力随自由度（节点数）变化，对照解析解"""
    n_nodes = np.array([r["n_nodes"] for r in conv_rows])
    disp = np.array([r["tip_deflection_mm"] for r in conv_rows])
    stress = np.array([r["max_mises_MPa"] for r in conv_rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    ax1.plot(n_nodes, disp, "-o", color="#2c5f8a", lw=1.8, label="FEA 挠度")
    ax1.axhline(analytical["delta"] * 1000, color="#c0392b", ls="--",
                label=f"解析解 {analytical['delta']*1000:.2f}mm")
    ax1.set_xlabel("节点数"); ax1.set_ylabel("端部挠度 (mm)")
    ax1.set_title("网格收敛性：挠度"); ax1.grid(alpha=0.3); ax1.legend(fontsize=10)

    ax2.plot(n_nodes, stress, "-o", color="#3a7d44", lw=1.8, label="FEA 最大应力")
    ax2.axhline(analytical["stress"] / 1e6, color="#c0392b", ls="--",
                label=f"梁理论截面应力 {analytical['stress']/1e6:.0f}MPa")
    ax2.set_xlabel("节点数"); ax2.set_ylabel("最大 Mises (MPa)")
    ax2.set_title("网格收敛性：应力（约束局部含应力集中）")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
