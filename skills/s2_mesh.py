"""
S2 参数化几何 + 网格 Skill（gmsh 真实网格引擎）
==================================================
输入：h(梁高), t(板厚), mesh_size(基准尺寸), refine(是否固定端加密)
输出：节点坐标、三角形单元连接、网格质量报告
封装的 Know-how：固定端加密区=1.5h、加密尺寸=基准1/2、质量门限检查
"""
import numpy as np
import gmsh
from .base import BaseSkill, ERR_MESH
import knowhow


class MeshingSkill(BaseSkill):
    skill_name = "s2_meshing"

    def _validate(self, params):
        super()._validate(params)
        for key in ("h", "t"):
            if key not in params:
                raise ValueError(f"缺少参数 {key}")

    def _execute(self, params):
        h = float(params["h"])
        t = float(params["t"])
        L = params.get("L", 1.0)
        base_size = float(params.get("mesh_size", knowhow.mesh_size_for(h)))
        refine = params.get("refine", True)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cantilever")
        rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, h)
        gmsh.model.occ.synchronize()

        fine = base_size * knowhow.MESH_STRATEGY["refine_ratio"]
        if refine:
            # 固定端局部加密场（Box 场 + Min 背景场）
            zone = knowhow.MESH_STRATEGY["refine_zone_length_h"] * h
            gmsh.model.mesh.field.add("Box", 1)
            gmsh.model.mesh.field.setNumber(1, "VIn", fine)
            gmsh.model.mesh.field.setNumber(1, "VOut", base_size)
            gmsh.model.mesh.field.setNumber(1, "XMin", -1e-9)
            gmsh.model.mesh.field.setNumber(1, "XMax", zone)
            gmsh.model.mesh.field.setNumber(1, "YMin", -1e-9)
            gmsh.model.mesh.field.setNumber(1, "YMax", h + 1e-9)
            gmsh.model.mesh.field.add("Min", 2)
            gmsh.model.mesh.field.setAsBackgroundMesh(2)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMin", fine)
            gmsh.option.setNumber("Mesh.MeshSizeMax", base_size)

        gmsh.option.setNumber("Mesh.MeshSizeMin", fine)
        gmsh.option.setNumber("Mesh.MeshSizeMax", base_size)
        gmsh.model.mesh.generate(2)

        # ---- 提取节点 ----
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.asarray(node_coords).reshape(-1, 3)
        tag2idx = {int(tag): i for i, tag in enumerate(node_tags)}
        points = node_coords[:, :2].copy()

        # ---- 提取三角形单元（2D type=2） ----
        elem_types, _, elem_conn = gmsh.model.mesh.getElements(dim=2)
        tris = None
        for etype, conn in zip(elem_types, elem_conn):
            if etype == 2:  # 3-node triangle
                conn = np.asarray(conn, dtype=int).reshape(-1, 3)
                tris = np.array([[tag2idx[c] for c in row] for row in conn])
                break
        gmsh.finalize()

        if tris is None or len(tris) == 0:
            return self.fail(ERR_MESH, "未生成三角形单元")

        quality = self._quality_report(points, tris)
        data = {
            "points": points,
            "tris": tris,
            "n_nodes": len(points),
            "n_elems": len(tris),
            "quality": quality,
            "base_size": base_size,
        }
        if not quality["pass"]:
            return self.fail(ERR_MESH, f"网格质量不合格: {quality['fail_reason']}", data=data)
        return self.success(data=data, msg=f"网格生成 {len(points)} 节点 / {len(tris)} 单元")

    @staticmethod
    def _quality_report(points, tris):
        """计算每个三角形最小内角与长宽比，按经验门限判定"""
        min_angles, aspects = [], []
        for tri in tris:
            p = points[tri]
            edges = np.roll(p, -1, axis=0) - p
            lens = np.linalg.norm(edges, axis=1)
            # 三边夹角（余弦定理）
            a, b, c = lens
            angles = []
            for i in range(3):
                u, v = edges[i], edges[(i + 1) % 3]
                cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30)
                angles.append(np.degrees(np.arccos(np.clip(-cosang, -1, 1))))
            min_angles.append(min(angles))
            aspects.append(lens.max() / (lens.min() + 1e-30))
        min_angles = np.array(min_angles)
        aspects = np.array(aspects)
        q = {
            "min_angle_deg": float(min_angles.min()),
            "median_min_angle": float(np.median(min_angles)),
            "max_aspect": float(aspects.max()),
            "median_aspect": float(np.median(aspects)),
        }
        reason = []
        if q["max_aspect"] > knowhow.MESH_QUALITY["max_aspect"]:
            reason.append(f"长宽比 {q['max_aspect']:.2f} > {knowhow.MESH_QUALITY['max_aspect']}")
        if q["min_angle_deg"] < 15.0:  # 三角形最小角经验红线
            reason.append(f"最小角 {q['min_angle_deg']:.1f}° < 15°")
        q["fail_reason"] = "; ".join(reason)
        q["pass"] = len(reason) == 0
        return q
