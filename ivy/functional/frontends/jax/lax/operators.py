# global
import ivy


def add(x, y):
    return ivy.add(x, y)


add.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def tan(x):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def concatenate(operands, dimension):
    return ivy.concat(operands, dimension)

def cholesky(x:[...,n,n], *, symmetrize_input=True):
    return ivy.cholesky(x, symmetrize_input=symmetrize_input)

def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


full.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def abs(x):
    return ivy.abs(x)


abs.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def sqrt(x):
    return ivy.sqrt(x)


sqrt.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
