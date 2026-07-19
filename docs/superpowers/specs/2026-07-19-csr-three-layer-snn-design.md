# CSR Three-Layer SNN Training Design

## Goal

Add a small BrainEvent CSR feedforward SNN training example under `training task/`.
The example should run on GPU for the intended demo, while keeping a small CPU
configuration available for automated smoke tests.

## Network

The model is a three-layer feedforward SNN:

```text
input spikes -> hidden1 LIF -> hidden2 LIF -> dense readout
```

The trainable layers are:

- `w01_data`: CSR weights from input neurons to hidden layer 1.
- `w12_data`: CSR weights from hidden layer 1 to hidden layer 2.
- `out_w`: dense readout weights from hidden layer 2 spike rate to classes.
- `out_b`: dense readout bias.

The CSR structures (`indices`, `indptr`, and `shape`) are fixed. Only CSR `data`
arrays are trained.

## Task

Use an online toy classification task. Each class owns a contiguous segment of
input neurons. For each batch and time step, random scores are class-biased, and
the top `active_count` input neurons are converted into binary spikes.

The model learns to predict the class label from the generated spike sequence.
There is no persistent dataset or train/test split.

## Forward Pass

For each time step:

```text
I1_t = BinaryArray(X_t) @ W01
S1_t = LIF(I1_t)
I2_t = BinaryArray(S1_t) @ W12
S2_t = LIF(I2_t)
Q2_t = Q2_{t-1} + S2_t
```

After `n_steps`:

```text
spike_rate2 = Q2 / n_steps
logits = spike_rate2 @ out_w + out_b
loss = cross_entropy(logits, labels)
```

Both hidden layers use the same surrogate spike function as the existing
training examples, with a custom JVP for differentiability through the hard
threshold.

## GPU Defaults

The script should default to a GPU-oriented size that is still modest:

- `n_inputs = 4096`
- `n_hidden1 = 2048`
- `n_hidden2 = 1024`
- `density01 = 0.002`
- `density12 = 0.004`
- `n_steps = 16`
- `batch_size = 16`
- `train_steps = 200`

The CLI should expose `--backend`, allowing explicit BrainEvent backend
selection such as `--backend cuda_raw` on GPU.

## Testing

Add a pytest smoke test that imports the sample module, builds small CSR
matrices, creates one toy batch, evaluates metrics, and runs one training step.
The test should assert:

- CSR shapes and nonzero counts match fixed fanout.
- Input and label shapes are correct.
- Loss is finite before and after one update.
- `w01_data`, `w12_data`, `out_w`, and `out_b` shapes are preserved.
- At least one training run with `train_steps=0` prints timing output.

The test configuration should be small enough for CPU.
