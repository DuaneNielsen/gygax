from typing import TypeVar, Container

import numpy as np
from jax import numpy as jnp

C = TypeVar('C')


def default_values(clazz: C) -> C:
    kwargs = {}
    for field_name, field_info in clazz.__dataclass_fields__.items():
        if issubclass(field_info.type, Container):
            kwargs[field_name] = default_values(field_info.type)
        else:
            kwargs[field_name] = jnp.array(jnp.zeros(1, dtype=field_info.type))
    return clazz(**kwargs)


def convert(pydataclass, clazz: C) -> C:
    kwargs = {}
    for field_name, field_info in clazz.__dataclass_fields__.items():

        if issubclass(field_info.type, Container):
            field_value = getattr(pydataclass, field_name)
            if field_value is not None:
                kwargs[field_name] = convert(field_value, field_info.type)
            else:
                kwargs[field_name] = default_values(field_info.type)
        elif issubclass(field_info.type, JaxStringArray):
            if type(pydataclass) is list:
                expanded_arrays = [JaxStringArray.str_to_uint8_array(getattr(s, field_name)) for s in pydataclass]
                kwargs[field_name] = jnp.stack(expanded_arrays)
            else:
                field_value = getattr(pydataclass, field_name)
                kwargs[field_name] = JaxStringArray.str_to_uint8_array(field_value)
        else:
            if type(pydataclass) is list:
                expanded_arrays = [getattr(x, field_name) for x in pydataclass]
                kwargs[field_name] = jnp.array(expanded_arrays, dtype=field_info.type)
            else:
                kwargs[field_name] = jnp.array(getattr(pydataclass, field_name), dtype=field_info.type)

    return clazz(**kwargs)


def fix_length(text: str, target_length: int, fill_char=" ") -> str:
    if len(text) > target_length:
        return text[:target_length]
    return text.ljust(target_length, fill_char)


class JaxStringArray:
    """
    A class to represent strings in jax
    """

    @staticmethod
    def str_to_uint8_array(text: str):
        text = fix_length(text, 20)
        # Convert string to bytes, then to uint8 array
        return jnp.array(list(text.encode('utf-8')), dtype=jnp.uint8)

    @staticmethod
    def uint8_array_to_str(arr):
        # Convert uint8 array to bytes, then to string
        arr_cpu = np.asarray(arr)
        return bytes(arr_cpu).decode('utf-8').strip()
