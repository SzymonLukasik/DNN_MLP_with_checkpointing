# Multi-Layer Perceptron with Checkpointing

This README introduces two implementations of a Multi-Layer Perceptron (MLP): the traditional approach and an optimized version using checkpointing.

## Introduction

The Multi-Layer Perceptron is a class of feedforward artificial neural network. While it's a powerful model, training deep MLPs can be memory-intensive. Checkpointing offers a way to manage memory consumption, especially when training on memory-constrained devices.

## Implementation Details

### 1. **Traditional Multi-Layer Perceptron (`Network`)**
The basic MLP implementation is straightforward, without any specific memory optimizations. It uses traditional forward and backward propagation algorithms.

Here's an example of how to initialize and use the MLP:
```python
nn = Network(layers=[64, 128, 64, 10])
nn.forward(x_train)
nn.backward(y_train)
```

### 2. **MLP with Checkpointing (`NetworkWithCheckpointing`)**

This advanced MLP version introduces checkpointing to optimize memory usage. This implementation:
- Saves outputs after activation for every n-th layer.
- Recalculates certain layers during the backward pass if their outputs were not saved.

Example of initializing and using the checkpointed MLP:
```python
nn_check = NetworkWithCheckpointing(layers=[64, 128, 64, 10], checkpoint_frequency=2)
nn_check.forward(x_train)
nn_check.backward(y_train)
```

Some specific methods in this class that manage the checkpointing mechanism include:
- `forward_up_to_layer`: Calculates forward pass up to a given layer.
- `get_cached_after_or_calculate`: Retrieves cached outputs or recalculates them if necessary.
- `cache_afters_during_forward_pass`: Manages the caching of output activations during the forward pass.
- `get_params_deltas_from_backward_pass`: Handles the backward pass, considering the cached values.

## Why Checkpointing?

By not saving every intermediate value during forward and backward passes and saving outputs after activation for every n-th layer, we reduce the memory footprint. However, we might have to recompute some values during the backward pass. The added computation time can be acceptable for the benefit of reduced memory usage.

## Testing and Verification

To ensure the checkpointing mechanism's correctness and efficiency, users are advised to:
1. Test for different network depths and checkpointing frequencies.
2. Monitor the cache usage during forward and backward passes.
3. Compare training time and accuracy between the traditional MLP and the checkpointed version.
4. Analyze memory savings due to checkpointing.

## Conclusion

These implementations provide both a basic understanding of the MLP and an advanced memory-optimized version with checkpointing. While the checkpointed version might incur additional computational overhead, the memory savings can be significant, especially for deep networks.

**Note**: Before running the code, ensure that the required dependencies and datasets (like `x_train`, `y_train`, etc.) are properly set up and imported.