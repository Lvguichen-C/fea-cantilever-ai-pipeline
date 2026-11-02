"""
BaseSkill —— 所有 Skill 的统一基类
==================================================
统一协议：
  输入：dict 参数
  输出：{"status": 错误码, "data": dict, "msg": str, "files": list}
错误码：
  0 成功；1 输入非法；2 网格失败；3 求解不收敛；4 约束违反；9 其他异常
"""
import traceback

# 状态码
OK = 0
ERR_INPUT = 1
ERR_MESH = 2
ERR_SOLVE = 3
ERR_CONSTRAINT = 4
ERR_OTHER = 9

CODE_NAME = {
    OK: "OK", ERR_INPUT: "INPUT_ERROR", ERR_MESH: "MESH_ERROR",
    ERR_SOLVE: "SOLVE_ERROR", ERR_CONSTRAINT: "CONSTRAINT_VIOLATION",
    ERR_OTHER: "UNKNOWN",
}


class BaseSkill:
    skill_name = "base"

    def run(self, params: dict) -> dict:
        """子类重写 _execute；这里统一做异常兜底和返回封装"""
        try:
            self._validate(params)
            result = self._execute(params)
            if "status" not in result:
                result["status"] = OK
            result.setdefault("data", {})
            result.setdefault("msg", CODE_NAME[result["status"]])
            result.setdefault("files", [])
            return result
        except Exception as exc:  # 任何异常都不让整条链路崩溃
            return {
                "status": ERR_OTHER,
                "data": {},
                "msg": f"{type(exc).__name__}: {exc}",
                "files": [],
                "trace": traceback.format_exc(),
            }

    def _validate(self, params):
        if not isinstance(params, dict):
            raise ValueError("params 必须是 dict")

    def _execute(self, params):
        raise NotImplementedError

    @staticmethod
    def success(data=None, msg="OK", files=None):
        return {"status": OK, "data": data or {}, "msg": msg, "files": files or []}

    @staticmethod
    def fail(code, msg, data=None, files=None):
        return {"status": code, "data": data or {}, "msg": msg, "files": files or []}
