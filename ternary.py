from enum import Enum
from typing import Union, Optional, List

import jax
import jax.numpy as jnp


"""Three-State Boolean Logic (0=FALSE, 1=TRUE, 2=NOT_DEFINED where NOT_DEFINED means "not in set")

Truth tables:

NOT: NOT(0)=1, NOT(1)=0, NOT(2)=2 (inverting doesn't put it in set)

AND/OR: If all NOT_DEFINED return NOT_DEFINED, else ignore NOT_DEFINED values:
AND: 1&1=1, 1&0=0, 0&1=0, 0&0=0, 2&2=2, 1&2=1, 0&2=0, 2&1=1, 2&0=0
OR:  1|1=1, 1|0=1, 0|1=1, 0|0=0, 2|2=2, 1|2=1, 0|2=0, 2|1=1, 2|0=0

ADD (composition): Copies RHS unless RHS is NOT_DEFINED:
1+1=1, 1+0=0, 0+1=1, 0+0=0, 1+2=1, 0+2=0, 2+1=1, 2+0=0, 2+2=2

Reductions:
ANY/ALL: Return NOT_DEFINED if all values are NOT_DEFINED
Otherwise treat as regular any()/all() with True=1, False=0, NotDefined=2
"""


class TernaryValue(Enum):
    FALSE = 0
    TRUE = 1
    NOT_SET = 2


class Ternary:
    def __init__(self, value: Optional[Union[bool, None, int, 'Ternary']] = None):
        if isinstance(value, Ternary):
            self.value = value.value
        elif isinstance(value, bool):
            self.value = TernaryValue.TRUE if value else TernaryValue.FALSE
        elif value is None:
            self.value = TernaryValue.NOT_SET
        elif isinstance(value, int):
            if value == 1:
                self.value = TernaryValue.TRUE
            elif value == 0:
                self.value = TernaryValue.FALSE
            else:
                self.value = TernaryValue.NOT_SET
        else:
            self.value = TernaryValue.NOT_SET

    def __and__(self, other: 'Ternary') -> 'Ternary':
        if not isinstance(other, Ternary):
            other = Ternary(other)

        # If both are NOT_SET, return NOT_SET
        if self.value == TernaryValue.NOT_SET and other.value == TernaryValue.NOT_SET:
            return Ternary(None)
        # If one is NOT_SET, return the other value
        elif self.value == TernaryValue.NOT_SET:
            return other
        elif other.value == TernaryValue.NOT_SET:
            return self
        # Regular AND logic for set values
        elif self.value == TernaryValue.FALSE or other.value == TernaryValue.FALSE:
            return Ternary(False)
        else:
            return Ternary(True)

    def __or__(self, other: 'Ternary') -> 'Ternary':
        if not isinstance(other, Ternary):
            other = Ternary(other)

        # If both are NOT_SET, return NOT_SET
        if self.value == TernaryValue.NOT_SET and other.value == TernaryValue.NOT_SET:
            return Ternary(None)
        # If one is NOT_SET, return the other value
        elif self.value == TernaryValue.NOT_SET:
            return other
        elif other.value == TernaryValue.NOT_SET:
            return self
        # Regular OR logic for set values
        elif self.value == TernaryValue.TRUE or other.value == TernaryValue.TRUE:
            return Ternary(True)
        else:
            return Ternary(False)

    def __invert__(self) -> 'Ternary':
        if self.value == TernaryValue.TRUE:
            return Ternary(False)
        elif self.value == TernaryValue.FALSE:
            return Ternary(True)
        else:
            return Ternary(None)  # Return NOT_SET for NOT_SET input

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ternary):
            other = Ternary(other)
        return self.value == other.value

    def __str__(self) -> str:
        return str(self.value.name)

    def __repr__(self) -> str:
        return f"Ternary({self.value.name})"


class TernaryArray:
    def __init__(self, value: Union[jnp.int8, Ternary, TernaryValue, List[Ternary], List[TernaryValue]]):
        if isinstance(value, jax.Array):
            if value.dtype == jnp.uint8:
                self.value = value
            else:
                TypeError(f"Unsupported type for value: {type(value)}")
        elif isinstance(value, List):
            if isinstance(value[0], TernaryValue):
                self.value = jnp.uint8(value)
            elif isinstance(value[0], Ternary):
                # Convert list of Ternary objects to list of their values
                self.value = jnp.uint8([tern.value for tern in value])
        else:
            if isinstance(value, Ternary):
                self.value = jnp.uint8(value.value)
            elif isinstance(value, TernaryValue):
                self.value = jnp.uint8(value)
            else:
                raise TypeError(f"Unsupported type for value: {type(value)}")

    def __repr__(self):
        return f"TernaryArray({self.value})"

    def __eq__(self, other):
        if isinstance(other, TernaryArray):
            return jnp.array_equal(self.value, other.value)
        return False

    def __invert__(self):
        return TernaryArray(jnp.where(self.value == 2, 2, 1 - self.value))

    def __and__(self, other):
        if not isinstance(other, TernaryArray):
            other = TernaryArray(other)

        # If both are 2, return 2
        both_undefined = jnp.logical_and(self.value == 2, other.value == 2)
        # Otherwise:
        # - If one is 2, use the other value
        # - If neither is 2, do regular AND
        result = jnp.where(both_undefined, 2,
                           jnp.where(self.value == 2, other.value,
                                     jnp.where(other.value == 2, self.value,
                                               self.value & other.value)))
        return TernaryArray(result)

    def __or__(self, other):
        if not isinstance(other, TernaryArray):
            other = TernaryArray(other)
        all_not_defined = jnp.all(
            jnp.logical_and(self.value == 2, other.value == 2)
        )
        return TernaryArray(jnp.where(all_not_defined, 2,
                                      jnp.where(self.value == 2, other.value,
                                                jnp.where(other.value == 2, self.value,
                                                          self.value | other.value))))

    def __add__(self, other):
        if not isinstance(other, TernaryArray):
            other = TernaryArray(other)
        return TernaryArray(jnp.where(other.value == 2, self.value, other.value))

    def all(self, axis=None, out=None, keepdims=False, *, where=None):
        """Implements three-state logical ALL reduction.

        Returns:
            - 2 (NOT_DEFINED) if all values are 2
            - 0 (False) if any non-2 value is 0
            - 1 (True) if all non-2 values are 1
        """
        is_undefined = self.value == 2
        all_undefined = jnp.all(is_undefined, axis=axis, keepdims=keepdims, where=where)

        # Mask out NOT_DEFINED values (treat them as True so they don't affect the result)
        masked_values = jnp.where(is_undefined, 1, self.value)
        regular_all = jnp.all(masked_values, axis=axis, keepdims=keepdims, where=where)

        result = jnp.where(all_undefined, jnp.uint8(2), regular_all)
        return TernaryArray(result)

    def any(self, axis=None, out=None, keepdims=False, *, where=None):
        """Implements three-state logical ANY reduction.

        Returns:
            - 2 (NOT_DEFINED) if all values are 2
            - 1 (True) if any non-2 value is 1
            - 0 (False) if all non-2 values are 0
        """
        is_undefined = self.value == 2
        all_undefined = jnp.all(is_undefined, axis=axis, keepdims=keepdims, where=where)

        # Mask out NOT_DEFINED values (treat them as False so they don't affect the result)
        masked_values = jnp.where(is_undefined, 0, self.value)
        regular_any = jnp.any(masked_values, axis=axis, keepdims=keepdims, where=where)

        result = jnp.where(all_undefined, jnp.uint8(2), regular_any)
        return TernaryArray(result)
