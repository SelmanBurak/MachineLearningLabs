"""
Dynamic DAG Computation Graph Engine
Student Engineer: Selman Burak Uygurtürk (Student ID: 72922)
Faculty: Applied Computer Science, Vistula University
Module: Lab 05 - Frameworks and Automatic Differentiation
"""

from __future__ import annotations
import math
from typing import Set, Tuple, Callable, Optional, Union, List


class GraphScalar:
    """
    Scalar DAG node storing primal numerical data, accumulated partial adjoint,
    predecessor dependencies, and local chain-rule derivative closure.
    """
    __slots__ = ('scalar', 'grad', '_predecessors', '_backward_rule', 'operator_str', 'symbol')

    def __init__(
        self,
        scalar: Union[float, int],
        predecessors: Tuple[GraphScalar, ...] = (),
        operator_str: str = '',
        symbol: Optional[str] = None
    ) -> None:
        self.scalar: float = float(scalar)
        self.grad: float = 0.0
        self._predecessors: Set[GraphScalar] = set(predecessors)
        self._backward_rule: Callable[[], None] = lambda: None
        self.operator_str: str = operator_str
        self.symbol: Optional[str] = symbol

    def __repr__(self) -> str:
        s = f" [{self.symbol}]" if self.symbol else ""
        return f"GraphScalar({self.scalar:.4f}, grad={self.grad:.4f}{s})"

    def __add__(self, other: Union[GraphScalar, float, int]) -> GraphScalar:
        rhs = other if isinstance(other, GraphScalar) else GraphScalar(other)
        out = GraphScalar(self.scalar + rhs.scalar, (self, rhs), 'Add')

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            rhs.grad += 1.0 * out.grad

        out._backward_rule = _backward
        return out

    def __radd__(self, other: Union[float, int]) -> GraphScalar:
        return self.__add__(other)

    def __mul__(self, other: Union[GraphScalar, float, int]) -> GraphScalar:
        rhs = other if isinstance(other, GraphScalar) else GraphScalar(other)
        out = GraphScalar(self.scalar * rhs.scalar, (self, rhs), 'Mul')

        def _backward() -> None:
            self.grad += rhs.scalar * out.grad
            rhs.grad += self.scalar * out.grad

        out._backward_rule = _backward
        return out

    def __rmul__(self, other: Union[float, int]) -> GraphScalar:
        return self.__mul__(other)

    def __pow__(self, p: Union[float, int]) -> GraphScalar:
        assert isinstance(p, (int, float)), "Power exponent must be numeric"
        out = GraphScalar(self.scalar ** p, (self,), f'Pow({p})')

        def _backward() -> None:
            self.grad += (p * (self.scalar ** (p - 1))) * out.grad

        out._backward_rule = _backward
        return out

    def relu(self) -> GraphScalar:
        v = self.scalar if self.scalar > 0.0 else 0.0
        out = GraphScalar(v, (self,), 'ReLU')

        def _backward() -> None:
            self.grad += (1.0 if self.scalar > 0.0 else 0.0) * out.grad

        out._backward_rule = _backward
        return out

    def sigmoid(self) -> GraphScalar:
        if self.scalar >= 0.0:
            val = 1.0 / (1.0 + math.exp(-self.scalar))
        else:
            val = math.exp(self.scalar) / (1.0 + math.exp(self.scalar))
        out = GraphScalar(val, (self,), 'Sigmoid')

        def _backward() -> None:
            self.grad += (val * (1.0 - val)) * out.grad

        out._backward_rule = _backward
        return out

    def log(self, eps: float = 1e-12) -> GraphScalar:
        val = max(self.scalar, eps)
        out = GraphScalar(math.log(val), (self,), 'Log')

        def _backward() -> None:
            self.grad += (1.0 / val) * out.grad

        out._backward_rule = _backward
        return out

    def __neg__(self) -> GraphScalar:
        return self * -1.0

    def __sub__(self, other: Union[GraphScalar, float, int]) -> GraphScalar:
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> GraphScalar:
        return GraphScalar(other) + (-self)

    def __truediv__(self, other: Union[GraphScalar, float, int]) -> GraphScalar:
        rhs = other if isinstance(other, GraphScalar) else GraphScalar(other)
        return self * (rhs ** -1.0)

    def backward(self) -> None:
        """
        Executes reverse-mode automatic differentiation:
        - Topologically sorts the computation DAG via depth-first post-order traversal.
        - Seeds root gradient to 1.0.
        - Evaluates local derivative rules in reverse topological order.
        - Accumulates adjoints in place using +=.
        """
        ordered: List[GraphScalar] = []
        visited: Set[GraphScalar] = set()

        def dfs(node: GraphScalar) -> None:
            if node not in visited:
                visited.add(node)
                for pred in node._predecessors:
                    dfs(pred)
                ordered.append(node)

        dfs(self)
        self.grad = 1.0

        for n in reversed(ordered):
            n._backward_rule()
