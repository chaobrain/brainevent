# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
import operator
from typing import Tuple, Union

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import JAXSparse
from brainevent._compatible_import import pallas as pl
from brainevent._coo import COO
from brainevent._csr import CSR
from brainevent._misc import _block_csr_tocoo, _nonzero_blocks, _block_csr_tocsr, estimate_block_size, csr_to_csc_index
from brainevent._typing import Data, Indptr, Index, MatrixShape

__all__ = [
    'BlockCSR',
]


@jax.tree_util.register_pytree_node_class
class BlockCSR(u.sparse.SparseMatrix):
    """
    Unit-aware Block-CSR sparse matrix.
    """
    __module__ = 'brainevent'

    data: Data
    indices: Index
    indptr: Indptr
    shape: MatrixShape

    ndim: int = property(lambda self: len(self.shape))
    num_blocks = property(lambda self: self.data.shape[0])
    block_size = property(lambda self: self.data.shape[1:])
    dtype = property(lambda self: self.data.dtype)
    nse: int = property(lambda self: self.indices.size * self.data.shape[1] * self.data.shape[2])

    def __init__(
        self,
        args: Tuple[Data, Index, Indptr],
        *,
        shape: MatrixShape,
    ):
        self.data, self.indices, self.indptr = map(u.math.asarray, args)

        super().__init__(args, shape=shape)

    def tree_flatten(self):
        return (self.data,), (self.indices, self.indptr, self.shape)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, = children
        indices, indptr, shape = aux_data
        return cls((data, indices, indptr), shape=shape)

    def _validate(self):
        _nblocks, n, m = self.data.shape
        nrows = self.indptr.shape[0] - 1
        assert self.indices.shape[0] == _nblocks
        assert len(self.shape) == 2
        assert self.shape[0] == n * nrows
        assert self.shape[1] % m == 0

    def tocsr(self) -> CSR:
        """
        Convert the BlockCSR matrix to CSR (Compressed Sparse Row) format.

        This method transforms the Block Compressed Sparse Row (BlockCSR) matrix into a CSR matrix.

        Returns
        -------
        CSR
            A CSR matrix containing the same data as the original BlockCSR matrix.

        Examples
        --------
        >>> block_csr_matrix = BlockCSR((data, indices, indptr), shape=(6, 6))
        >>> csr_matrix = block_csr_matrix.tocsr()

        """
        self._validate()
        data, indices, indptr = _block_csr_tocsr(self.data, self.indices, self.indptr, self.shape)
        return CSR((data, indices, indptr), shape=self.shape)

    def tocoo(self):
        """
        Convert the BlockCSR matrix to COO (Coordinate) format.

        This method transforms the Block Compressed Sparse Row (BlockCSR) matrix into a COO matrix,
        which stores sparse data as a collection of (row, column, value) triplets.

        Returns
        -------
        COO
            A COO matrix containing the same data as the original BlockCSR matrix.

        See Also
        --------
        _block_csr_to_coo : Internal function that converts BlockCSR row/column indices to COO format
        COO : The Coordinate sparse matrix class

        Examples
        --------
        >>> block_csr_matrix = BlockCSR((data, indices, indptr), shape=(3, 4))
        >>> coo_matrix = block_csr_matrix.tocoo()
        """
        self._validate()
        _, n, m = self.data.shape
        with jax.ensure_compile_time_eval():
            pre_ids, post_ids = _block_csr_tocoo(n, m, self.shape[0], self.nse, self.indices, self.indptr)
        return COO((self.data.reshape(-1), pre_ids, post_ids), shape=self.shape)

    def todense(self) -> jax.Array:  # BUG: when do todense() after doing fromdense(),  it will raise an error
        self._validate()
        return _sdd_todense(self)

    def todense_new(self):
        '''
        Since the underlying logic of csr_todense is to convert the csr to coo,
        and using coo_todense to convert the coo to dense,
        same like that, we can first convert block_csr to coo,
        and then directly use the coo_todense to convert the block_csr to dense.
        '''
        self._validate()
        return self.tocoo().todense()

    @classmethod
    def fromdense(cls, dense: jax.Array, *, block_size: Tuple[int, int] = None) -> 'BlockCSR':
        """
        Create a BlockCSR matrix from a dense matrix.

        This method converts a dense matrix to a BlockCSR format.

        Parameters
        -----------
        dense : array_like
            The dense matrix to be converted to BlockCSR format.
        block_size : tuple
            The size of each block in the BlockCSR matrix.

        Returns
        --------
        BlockCSR
            A new BlockCSR matrix object created from the input dense matrix.
        """
        if dense.ndim != 2:
            raise ValueError("Cannot convert a 1d sparse array to block_csr format")

        N, M = dense.shape
        if block_size is None:
            csr = CSR.fromdense(dense)
            block_size = estimate_block_size(csr)
        n, m = block_size

        if n < 1 or m < 1 or N % n != 0 or M % m != 0:
            raise ValueError(f"Invalid block size: {block_size} for matrix shape {dense.shape}. "
                             "The block size n and m must be positive, and the shape of the "
                             "dense matrix must be divisible by the block size.")

        nonzero_blocks, indices, indptr = _nonzero_blocks(dense, block_size)
        return BlockCSR((nonzero_blocks, indices, indptr), shape=(N, M))

    def with_data(self, data: Data) -> 'BlockCSR':
        """
        Create a new BlockCSR matrix with updated data while keeping the same structure.

        This method creates a new BlockCSR matrix instance with the provided data,
        maintaining the original indices, indptr, and shape.

        Parameters
        -----------
        data : Data
            The new data array to replace the existing data in the BlockCSR matrix.
            It must have the same shape, dtype, and unit as the original data.

        Returns
        --------
        BlockCSR
            A new BlockCSR matrix instance with updated data and the same structure as the original.

        Raises
        -------
        AssertionError
            If the shape, dtype, or unit of the new data doesn't match the original data.
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return BlockCSR((data, self.indices, self.indptr), shape=self.shape)

    def transpose(self, axes=None) -> 'BlockCSR':
        """
        Transpose the BlockCSR matrix.

        This method returns the transpose of the BlockCSR matrix as a BlockCSC matrix.

        Parameters
        -----------
        axes : None
            This parameter is not used and must be None. Included for compatibility
            with numpy's transpose function signature.

        Returns
        --------
        BlockCSR
            The transpose of the BlockCSR matrix.

        Raises
        -------
        AssertionError
            If axes is not None, as this implementation doesn't support custom axis ordering.
        """
        assert axes is None, "transpose does not support axes argument."
        data = self.data
        indices = self.indices
        indptr = self.indptr
        N, M = self.shape
        n, m = data.shape[1:]  # block size
        n_block_rows = indptr.shape[0] - 1
        n_block_cols = M // m

        inptr, indices, positions = csr_to_csc_index(indptr, indices, shape=(n_block_rows, n_block_cols))
        new_data = jnp.transpose(data, (0, 2, 1))  # (n_blocks, m, n)
        new_data = new_data[positions]
        new_shape = (self.shape[1], self.shape[0])
        return BlockCSR((new_data, indices, inptr), shape=new_shape)

    def _unitary_op(self, op) -> 'BlockCSR':
        """
        Apply a unary operation to the data of the BlockCSR matrix.

        This method applies a given unary operation to the data array of the BlockCSR matrix.

        Parameters
        ----------
        op : callable
            A unary operation to apply to the data array (e.g., abs, neg, pos).

        Returns
        -------
        BlockCSR
            A new BlockCSR matrix with the result of applying the operation to its data.
        """
        return BlockCSR((op(self.data), self.indices, self.indptr), shape=self.shape)

    def __abs__(self):
        return self._unitary_op(operator.abs)

    def __neg__(self):
        return self._unitary_op(operator.neg)

    def __pos__(self):
        return self._unitary_op(operator.pos)

    def _binary_op(self, other, op) -> 'BlockCSR':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(dense, other)

        if isinstance(other, BlockCSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return BlockCSR(
                    (op(self.data, other.data), self.indices, self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return BlockCSR(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )

        elif other.ndim == 2 and other.shape == self.block_size:
            return BlockCSR(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape,
            )

        else:
            raise ValueError(f"Unsupported operation: {op} between BlockCSR and {type(other)}")

    def __mul__(self, other: Data):
        return self._binary_op(other, operator.mul)

    def __div__(self, other: Data):
        return self._binary_op(other, operator.truediv)

    def __truediv__(self, other):
        return self.__div__(other)

    def __add__(self, other):
        return self._binary_op(other, operator.add)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub)

    def __mod__(self, other):
        return self._binary_op(other, operator.mod)

    def _binary_rop(self, other, op) -> 'BlockCSR':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(other, dense)

        if isinstance(other, BlockCSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return BlockCSR(
                    (op(other.data, self.data), other.indices, other.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return BlockCSR(
                (op(other, self.data), self.indices, self.indptr),
                shape=self.shape
            )

        elif other.ndim == 2 and other.shape == self.block_size:
            return BlockCSR(
                (op(other, self.data), self.indices, self.indptr),
                shape=self.shape,
            )

        else:
            raise ValueError(f"Unsupported operation: {op} between BlockCSR and {type(other)}")

    def __rmul__(self, other: Data):
        return self._binary_rop(other, operator.mul)

    def __rdiv__(self, other: Data):
        return self._binary_rop(other, operator.truediv)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __radd__(self, other):
        return self._binary_rop(other, operator.add)

    def __rsub__(self, other):
        return self._binary_rop(other, operator.sub)

    def __rmod__(self, other):
        return self._binary_rop(other, operator.mod)

    def solve(self, b: Union[jax.Array, u.Quantity], tol=1e-6, reorder=1) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system Ax = b where A is the sparse matrix.

        This method uses JAX's sparse solver to solve the equation Ax = b,
        where A is the current sparse matrix and b is the right-hand side vector.

        Parameters
        ----------
        b : array_like
            The right-hand side vector of the linear system.
        tol : Tolerance to decide if singular or not. Defaults to 1e-6.
        reorder : The reordering scheme to use to reduce fill-in. No reordering if
            ``reorder=0``. Otherwise, symrcm, symamd, or csrmetisnd (``reorder=1,2,3``),
            respectively. Defaults to symrcm.

        Returns
        -------
        x : jax.Array or u.Quantity
            The solution vector x that satisfies Ax = b.
        """
        csr = self.tocsr()
        return csr.solve(b, tol, reorder)

    def __matmul__(self, other) -> jax.Array:
        self._validate()
        if other.ndim == 2:
            return sdd_matmul(self, other)
        elif other.ndim == 1:
            raise NotImplementedError
            # For vector multiplication, we can use the dense representation
            # to avoid the overhead of the sparse matrix multiplication.
            return self.todense() @ other
        else:
            raise ValueError(f"Unsupported operation: BlockCSR @ {type(other)}. "
                             "Only support 2D array or 1D vector multiplication.")

    def __rmatmul__(self, other):
        # TODO: using tocsr VS manually implement the XLA kernel?
        # For example, block the other and do calculation?
        return other @ self.tocsr()  # lazy implementation for now, plan to implement the XLA kernel


def _sdd_todense(mat: BlockCSR) -> jax.Array:
    _, n, m = mat.data.shape
    nrows = mat.shape[0] // n
    unit = u.get_unit(mat.data)
    blocks = u.get_mantissa(mat.data)

    def i_body(i_row, out):  # each row
        def j_body(x):  # each block in the row
            i_block, val = x
            i_col = mat.indices[i_block]
            val = jax.lax.dynamic_update_slice(val, blocks[i_block], (i_row * n, i_col * m))
            return i_block + 1, val

        return jax.lax.while_loop(
            lambda x: x[0] < mat.indptr[i_row + 1],
            j_body,
            (mat.indptr[i_row], out)
        )[1]

    dense = jax.lax.fori_loop(0, nrows, i_body, jnp.zeros(mat.shape, mat.dtype))
    return u.maybe_decimal(u.Quantity(dense, unit=unit))


def _check_shape_consistency(x, y):
    assert isinstance(y, jax.Array), f"Only support jax.Array. But got unsupported type {type(y)}"
    assert x.ndim == y.ndim == 2
    assert x.shape[1] == y.shape[0], f"Dimension mismatch: {x.shape} @ {y.shape}"


def _sdd_kernel(
    x_ref,  # [n_blocks, bm, bn]
    indices_ref,  # [n_block]
    indptr_ref,  # [n_rows + 1]
    y_ref,  # [n, k]
    o_ref,  # [m, k]
    *,
    bm: int,
    bn: int,
    bk: int,
):
    i_m = pl.program_id(axis=0)
    i_k = pl.program_id(axis=1)
    i_start = indptr_ref[i_m]
    i_end = indptr_ref[i_m + 1]

    def body(x):
        val, i_block = x
        i_x_col = indices_ref[i_block]
        block = pl.load(x_ref, (i_block, pl.dslice(None), pl.dslice(None)))  # [bm, bn]
        chunk = pl.load(y_ref, (pl.dslice(i_x_col * bn, bn), pl.dslice(i_k * bk, bk)))  # [bn, bk]
        return val + jnp.dot(block, chunk).astype(o_ref.dtype), i_block + 1

    acc = jax.lax.while_loop(
        lambda x: x[1] < i_end,
        body,
        (jnp.zeros([bm, bk], dtype=o_ref.dtype), i_start)
    )[0]
    pl.store(o_ref, (pl.dslice(bm * i_m, bm), pl.dslice(bk * i_k, bk)), acc)  # [bm, bk]


@functools.partial(jax.jit, static_argnames=["debug", 'interpret', 'block_size'])
def sdd_matmul(
    mat1: BlockCSR,
    mat2: jax.Array,
    *,
    debug: bool = False,
    interpret: bool = False,
    block_size: int = 256,
) -> jax.Array:
    _check_shape_consistency(mat1, mat2)

    # shape and dtype
    m, n, k = mat1.shape[0], mat1.shape[1], mat2.shape[1]
    _, bm, bn = mat1.data.shape
    dtype = jnp.result_type(mat1.dtype, mat2.dtype)

    # kernel
    fn = pl.pallas_call(
        functools.partial(_sdd_kernel, bm=bm, bn=bn, bk=block_size),
        out_shape=jax.ShapeDtypeStruct(shape=(m, k), dtype=dtype),
        grid=(pl.cdiv(m, bm), pl.cdiv(k, block_size)),
        debug=debug,
        interpret=interpret
    )

    # call
    unita = u.get_unit(mat1.data)
    unitb = u.get_unit(mat2)
    blocks = u.get_mantissa(mat1.data)
    r = fn(blocks, mat1.indices, mat1.indptr, u.get_mantissa(mat2))
    return u.maybe_decimal(u.Quantity(r, unit=unita * unitb))


def native_sdd_matmul(
    mat1: BlockCSR,
    mat2: jax.Array,
):
    _check_shape_consistency(mat1, mat2)

    dtype = jnp.result_type(mat1.dtype, mat2.dtype)
    _, n, m = mat1.data.shape

    nrows = mat1.shape[0] // n

    def i_body(i):  # each row
        def k_body(x):
            i_block, val = x
            i_col = mat1.indices[i_block]
            chunk = jax.lax.dynamic_slice(mat2, [i_col * m, 0], (m, mat2.shape[1]))  # [m, mat2.shape[1]]
            block = blocks[i_block]
            return i_block + 1, val + block.dot(chunk)

        acc = jax.lax.while_loop(
            lambda x: x[0] < mat1.indptr[i + 1],
            k_body,
            (mat1.indptr[i], jnp.zeros((n, mat2.shape[1]), dtype=jnp.float32))
        )[1]
        return acc.astype(dtype)

    unita = u.get_unit(mat1.data)
    unitb = u.get_unit(mat2)
    blocks = u.get_mantissa(mat1.data)
    mat2 = u.get_mantissa(mat2)

    out = jax.vmap(i_body)(jnp.arange(nrows)).reshape((mat1.shape[0], mat2.shape[1]))
    return u.maybe_decimal(u.Quantity(out, unit=unita * unitb))


def sample_sparse_matrix(
    m,
    n,
    bm,
    bn,
    *,
    sparse_prob=0.2,
    dtype=jnp.float32
) -> BlockCSR:
    num_rows = m // bm  # number of rows in the Block-ELL matrix
    num_cols = n // bn  # number of columns in the Block-ELL matrix
    blocks_per_row = np.random.binomial(num_cols, sparse_prob,
                                        size=[num_rows])  # [n_rows], number of data in each row
    num_blocks = blocks_per_row.sum()
    blocks = np.random.randn(num_blocks, bm, bn).astype(dtype)  # [n_blocks, bm, bk], block values

    # [n_rows + 1], row pointers
    indptr = np.zeros(num_rows + 1, dtype=np.int32)  # [n_rows + 1], row pointers
    indptr[1:] = np.cumsum(blocks_per_row)

    # [n_block], block indices
    indices = []
    for i in range(num_rows):
        indices.extend(np.random.choice(num_cols, blocks_per_row[i], replace=False))
    indices = jnp.array(indices)  # [n_rows, max_num_blocks_per_row, 2], block indices

    return BlockCSR((jnp.asarray(blocks), jnp.asarray(indptr), jnp.asarray(indices)), shape=(m, n))
