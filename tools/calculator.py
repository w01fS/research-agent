import ast
import operator as op
from typing import Any, Union
from .base import Tool

# Mapping AST operator types to Python operators
_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}

_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


class CalculatorTool(Tool):
    name = "calculator"

    def run(self, input_data: str) -> Union[float, str]:
        """
        Evaluates arithmetic expressions safely using AST.
        """
        try:
            expr = ast.parse(input_data, mode="eval").body
            return self._eval(expr)
        except Exception as e:
            return f"Calculator error: {str(e)}"

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            op_type = type(node.op)
            if op_type in _BIN_OPS:
                return _BIN_OPS[op_type](left, right)
            else:
                raise ValueError(f"Unsupported operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            op_type = type(node.op)
            if op_type in _UNARY_OPS:
                return _UNARY_OPS[op_type](operand)
            else:
                raise ValueError(f"Unsupported unary operator: {op_type}")
        else:
            raise ValueError(f"Unsupported expression: {node}")