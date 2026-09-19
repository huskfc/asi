"""Safe softmax and log-softmax implementations immune to XLA recomputation NaN."""

import jax
import jax.numpy as jnp


def safe_softmax(logits: jax.Array, temperature: float = 1.0) -> jax.Array:
    """Softmax with double-centering to prevent XLA recomputation NaN.

    The double-centering construction:
    1. centered = logits - stop_gradient(max(logits))
    2. scaled = centered / temperature
    3. scaled_centered = scaled - stop_gradient(max(scaled))
    4. softmax(scaled_centered)

    This is immune to XLA fusing recomputation because:
    - The first centering ensures max(centered) = 0 exactly
    - The scaled values are all <= 0 (for positive temperature)
    - The second centering is numerically a no-op (max(scaled) = 0)
    - Gradients are preserved via stop_gradient on the max operations
    """
    centered = logits - jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    scaled = centered / temperature
    scaled_centered = scaled - jax.lax.stop_gradient(jnp.max(scaled, axis=-1, keepdims=True))
    return jax.nn.softmax(scaled_centered)


def safe_log_softmax(logits: jax.Array, temperature: float = 1.0) -> jax.Array:
    """Log-softmax with double-centering to prevent XLA recomputation NaN."""
    centered = logits - jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    scaled = centered / temperature
    scaled_centered = scaled - jax.lax.stop_gradient(jnp.max(scaled, axis=-1, keepdims=True))
    return jax.nn.log_softmax(scaled_centered)


def safe_softmax_with_logits(logits: jax.Array, temperature: float = 1.0) -> tuple[jax.Array, jax.Array]:
    """Return (probs, log_probs) both computed with double-centering."""
    centered = logits - jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    scaled = centered / temperature
    scaled_centered = scaled - jax.lax.stop_gradient(jnp.max(scaled, axis=-1, keepdims=True))
    probs = jax.nn.softmax(scaled_centered)
    log_probs = jax.nn.log_softmax(scaled_centered)
    return probs, log_probs
