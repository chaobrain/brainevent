# brainevent

Event-driven sparse linear algebra for spiking neural network simulation: weight-matrix representations and the operators that act on event (spike) vectors.

## Language

### Representations

**Data representation**:
A weight matrix `W` of shape `(num_pre, num_post)` in one of several storage formats, sharing one abstract base (`DataRepresentation`) and a common API contract. Carries a *buffer registry* for cached auxiliary structures.
_Avoid_: matrix format, sparse type (when referring to the abstraction)

**Compressed-sparse representation**:
A data representation storing explicit `(data, indices, indptr)` arrays — `CSR` (row-compressed) and `CSC` (column-compressed). Base: `CompressedSparseData`.

**Fixed-number connection**:
A data representation where every pre- (or post-) synaptic neuron has the *same fixed count* of connections, stored ELL-style. `FixedNumPerPre` mirrors `CSR`; `FixedNumPerPost` mirrors `CSC`. Base: `FixedNumConn`.
_Avoid_: ELL matrix, fixed fan-in/out (in prose; use the class concept)

**JIT-connectivity matrix**:
A data representation whose connections are *generated procedurally on demand* from a probability and a random seed, never materialised. Weights are a scalar or a distribution's parameters, not a per-synapse array. Families: scalar / normal / uniform, each in an `R` (row) and `C` (column) orientation. Base: `JITCMatrix`.
_Avoid_: random matrix, on-the-fly matrix

**Event representation**:
A spike/event *vector* (e.g. `BinaryArray`, `BitPackedBinary`) consumed by a data representation's matmul. A **separate** hierarchy (`EventRepresentation`), not a data representation.
_Avoid_: spike matrix, event array (when meaning the class)

### Protocols

**yw_to_w protocol**:
The per-synapse product `w * y[index]` returned over a representation's stored connections (`yw_to_w` indexes `y` by row/pre; `yw_to_w_transposed` by column/post). Used by `brainscale` for eligibility/gradient propagation.
_Avoid_: outer product, gradient kernel

**Buffer**:
A named, lazily-populated auxiliary array cached on a representation (e.g. a fixed-num connection's column-major mirror under key `'csc'`). Registered via `register_buffer`, surfaced through the `buffers` property.
_Avoid_: cache, attribute
