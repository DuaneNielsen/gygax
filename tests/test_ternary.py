from ternary import TernaryArray
import jax.numpy as jnp

def test_ternary_array_not():
    """Test NOT operation according to truth table NOT(0)=1, NOT(1)=0, NOT(2)=2"""
    cases = [
        ([0], [1]),
        ([1], [0]),
        ([2], [2]),
        ([0, 1, 2], [1, 0, 2])
    ]
    for input, expected in cases:
        arr = TernaryArray(jnp.uint8(input))
        result = ~arr
        assert jnp.array_equal(result.value, jnp.uint8(expected)), \
            f"Failed: NOT({input}) should be {expected}, got {result.value}"

def test_ternary_array_and():
    """Test AND operation according to truth table:
    1&1=1, 1&0=0, 0&1=0, 0&0=0, 2&2=2, 1&2=1, 0&2=0, 2&1=1, 2&0=0"""
    cases = [
        ([1], [1], [1]),  # 1&1=1
        ([1], [0], [0]),  # 1&0=0
        ([0], [1], [0]),  # 0&1=0
        ([0], [0], [0]),  # 0&0=0
        ([2], [2], [2]),  # 2&2=2
        ([1], [2], [1]),  # 1&2=1
        ([0], [2], [0]),  # 0&2=0
        ([2], [1], [1]),  # 2&1=1
        ([2], [0], [0]),  # 2&0=0
        # Test array combinations
        ([2, 2, 2], [2, 2, 2], [2, 2, 2]),  # All NOT_DEFINED
        ([1, 0, 2], [1, 1, 2], [1, 0, 2])   # Mixed case
    ]
    for a, b, expected in cases:
        arr1 = TernaryArray(jnp.uint8(a))
        arr2 = TernaryArray(jnp.uint8(b))
        result = arr1 & arr2
        assert jnp.array_equal(result.value, jnp.uint8(expected)), \
            f"Failed: {a} AND {b} should be {expected}, got {result.value}"

def test_ternary_array_or():
    """Test OR operation according to truth table:
    1|1=1, 1|0=1, 0|1=1, 0|0=0, 2|2=2, 1|2=1, 0|2=0, 2|1=1, 2|0=0"""
    cases = [
        ([1], [1], [1]),  # 1|1=1
        ([1], [0], [1]),  # 1|0=1
        ([0], [1], [1]),  # 0|1=1
        ([0], [0], [0]),  # 0|0=0
        ([2], [2], [2]),  # 2|2=2
        ([1], [2], [1]),  # 1|2=1
        ([0], [2], [0]),  # 0|2=0
        ([2], [1], [1]),  # 2|1=1
        ([2], [0], [0]),  # 2|0=0
        # Test array combinations
        ([2, 2, 2], [2, 2, 2], [2, 2, 2]),  # All NOT_DEFINED
        ([1, 0, 2], [0, 1, 2], [1, 1, 2])   # Mixed case
    ]
    for a, b, expected in cases:
        arr1 = TernaryArray(jnp.uint8(a))
        arr2 = TernaryArray(jnp.uint8(b))
        result = arr1 | arr2
        assert jnp.array_equal(result.value, jnp.uint8(expected)), \
            f"Failed: {a} OR {b} should be {expected}, got {result.value}"

def test_ternary_array_add():
    """Test ADD operation according to truth table:
    1+1=1, 1+0=0, 0+1=1, 0+0=0, 1+2=1, 0+2=0, 2+1=1, 2+0=0, 2+2=2"""
    cases = [
        ([1], [1], [1]),  # 1+1=1
        ([1], [0], [0]),  # 1+0=0
        ([0], [1], [1]),  # 0+1=1
        ([0], [0], [0]),  # 0+0=0
        ([1], [2], [1]),  # 1+2=1
        ([0], [2], [0]),  # 0+2=0
        ([2], [1], [1]),  # 2+1=1
        ([2], [0], [0]),  # 2+0=0
        ([2], [2], [2]),  # 2+2=2
        # Test array combinations
        ([1, 0, 2], [2, 2, 2], [1, 0, 2]),  # RHS is all NOT_DEFINED
        ([2, 2, 2], [1, 0, 2], [1, 0, 2])   # Mixed case
    ]
    for a, b, expected in cases:
        arr1 = TernaryArray(jnp.uint8(a))
        arr2 = TernaryArray(jnp.uint8(b))
        result = arr1 + arr2
        assert jnp.array_equal(result.value, jnp.uint8(expected)), \
            f"Failed: {a} + {b} should be {expected}, got {result.value}"

def test_ternary_array_reductions():
    """Test ANY/ALL reductions: NOT_DEFINED if all NOT_DEFINED, else regular any/all"""
    # Test ANY
    any_cases = [
        ([2, 2, 2], 2),       # All NOT_DEFINED -> NOT_DEFINED
        ([0, 1, 2], 1),       # Has TRUE -> TRUE
        ([0, 0, 2], 0),       # Only FALSE and NOT_DEFINED -> FALSE
        ([0, 2, 2], 0)        # Only FALSE and NOT_DEFINED -> FALSE
    ]
    for arr, expected in any_cases:
        result = TernaryArray(jnp.uint8(arr)).any()
        assert result.value == expected, \
            f"Failed: any({arr}) should be {expected}, got {result.value}"

    # Test ALL
    all_cases = [
        ([2, 2, 2], 2),       # All NOT_DEFINED -> NOT_DEFINED
        ([1, 1, 2], 1),       # All TRUE or NOT_DEFINED -> TRUE
        ([0, 1, 2], 0),       # Has FALSE -> FALSE
        ([1, 2, 2], 1)        # All TRUE or NOT_DEFINED -> TRUE
    ]
    for arr, expected in all_cases:
        result = TernaryArray(jnp.uint8(arr)).all()
        assert result.value == expected, \
            f"Failed: all({arr}) should be {expected}, got {result.value}"


def test_ternary_array_all():
    """Test ALL reduction according to three-state logic:
    - Returns 2 if all values are 2
    - Returns 0 if any non-2 value is 0
    - Returns 1 if all non-2 values are 1
    """
    cases = [
        # Single value cases
        ([0], 0),  # Single False
        ([1], 1),  # Single True
        ([2], 2),  # Single NOT_DEFINED

        # All same value cases
        ([0, 0, 0], 0),  # All False
        ([1, 1, 1], 1),  # All True
        ([2, 2, 2], 2),  # All NOT_DEFINED

        # Mixed with NOT_DEFINED
        ([1, 2, 1], 1),  # All non-2 are True
        ([1, 2, 0], 0),  # Contains False
        ([0, 2, 0], 0),  # All non-2 are False

        # Edge cases
        ([2, 2, 1], 1),  # Mostly NOT_DEFINED
        ([2, 0, 2], 0),  # NOT_DEFINED with False
    ]

    for values, expected in cases:
        arr = TernaryArray(jnp.uint8(values))
        result = arr.all()
        assert jnp.array_equal(result.value, jnp.uint8(expected)), \
            f"Failed: all({values}) should be {expected}, got {result.value}"

    # Test with axis parameter
    arr = TernaryArray(jnp.uint8([[1, 2, 1],
                                  [1, 1, 2],
                                  [2, 2, 2]]))
    # Test along axis 0
    result = arr.all(axis=0)
    expected = [1, 1, 1]  # Each column has all 1s (ignoring 2s)
    assert jnp.array_equal(result.value, jnp.uint8(expected)), \
        f"Failed: all(axis=0) should be {expected}, got {result.value}"

    # Test along axis 1
    result = arr.all(axis=1)
    expected = [1, 1, 2]  # Last row is all 2s
    assert jnp.array_equal(result.value, jnp.uint8(expected)), \
        f"Failed: all(axis=1) should be {expected}, got {result.value}"


def test_ternary_array_any():
    """Test ANY reduction according to three-state logic:
    - Returns 2 if all values are 2
    - Returns 1 if any non-2 value is 1
    - Returns 0 if all non-2 values are 0
    """
    cases = [
        # Single value cases
        ([0], 0),  # Single False
        ([1], 1),  # Single True
        ([2], 2),  # Single NOT_DEFINED

        # All same value cases
        ([0, 0, 0], 0),  # All False
        ([1, 1, 1], 1),  # All True
        ([2, 2, 2], 2),  # All NOT_DEFINED

        # Mixed with NOT_DEFINED
        ([0, 2, 1], 1),  # Contains True
        ([0, 2, 0], 0),  # All non-2 are False
        ([1, 2, 0], 1),  # Contains both True and False

        # Edge cases
        ([2, 2, 1], 1),  # Mostly NOT_DEFINED with True
        ([2, 0, 2], 0),  # NOT_DEFINED with False
    ]

    for values, expected in cases:
        arr = TernaryArray(jnp.uint8(values))
        result = arr.any()
        assert jnp.array_equal(result.value, jnp.uint8(expected)), \
            f"Failed: any({values}) should be {expected}, got {result.value}"

    # Test with axis parameter
    arr = TernaryArray(jnp.uint8([[0, 2, 0],
                                  [0, 1, 2],
                                  [2, 2, 2]]))
    # Test along axis 0
    result = arr.any(axis=0)
    expected = [0, 1, 0]  # Middle column has a 1
    assert jnp.array_equal(result.value, jnp.uint8(expected)), \
        f"Failed: any(axis=0) should be {expected}, got {result.value}"

    # Test along axis 1
    result = arr.any(axis=1)
    expected = [0, 1, 2]  # Last row is all 2s
    assert jnp.array_equal(result.value, jnp.uint8(expected)), \
        f"Failed: any(axis=1) should be {expected}, got {result.value}"