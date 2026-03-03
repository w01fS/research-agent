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
    description = "Evaluates arithmetic expressions. Input should be a string expression like '2+2'."

    def run(self, input_data: Union[str, Dict]) -> Union[float, str]:
        """
        Evaluates arithmetic expressions safely using AST.
        """
        if isinstance(input_data, dict):
            # Try to find the expression in common keys
            expr_val = (
                input_data.get("expression") or 
                input_data.get("input") or 
                input_data.get("query") or 
                input_data.get("expr") or
                input_data.get("q")
            )
            if expr_val:
                input_data = str(expr_val)
            else:
                # If no known key, just try the first value if there's only one key
                values = list(input_data.values())
                if len(values) == 1:
                    input_data = str(values[0])
                else:
                    return f"Calculator error: Could not find expression in input dict {input_data}"

        try:
            expr = ast.parse(str(input_data), mode="eval").body
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