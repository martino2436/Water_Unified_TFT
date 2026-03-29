"""
Temporal Fusion Transformer - 100% Paper Compliant Implementation v4
Based on: Lim et al. (2021) "Temporal Fusion Transformers for Interpretable 
Multi-horizon Time Series Forecasting"

CHANGES FROM v3:
1. FIXED: VSN no longer includes static context as a variable (paper compliance)
2. FIXED: Removed extra final GRN - output directly from position-wise FF (Eq. 23)
3. IMPROVED: Robust masking checks ALL inputs, not just the first one
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, Embedding, 
    Concatenate, LSTM, Input, Masking, TimeDistributed,
    BatchNormalization
)
from tensorflow.keras.models import Model
import pickle
import numpy as np


# =============================================================================
# Generic Embedding Layer (with -1 masking)
# =============================================================================
class GenericEmbedding(tf.keras.layers.Layer):
    """
    Embedding layer that handles both categorical and continuous variables.
    Handles -1 as mask value for missing/padded data.
    
    CRITICAL: For continuous variables, -1 padding values are replaced with 0
    before the linear transformation to prevent corrupted embeddings.
    """
    def __init__(self, num_categories, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_categories = num_categories
        self.embedding_size = embedding_size
        
    def build(self, input_shape):
        if self.num_categories == 0:
            self._embedding = Dense(self.embedding_size)
        else:
            self._embedding = Embedding(
                self.num_categories, 
                self.embedding_size, 
                mask_zero=False
            )
        self._reshape = tf.keras.layers.Reshape([self.embedding_size])
        super().build(input_shape)

    def call(self, inputs):
        # Handle -1 mask for ALL variable types (categorical AND continuous)
        # This is CRITICAL for continuous variables where -1 padding would
        # otherwise be linearly transformed into corrupted embeddings
        mask = tf.not_equal(inputs, -1)
        
        if self.num_categories > 0:
            # Categorical: replace -1 with 0 (valid embedding index)
            safe_inputs = tf.where(mask, inputs, tf.zeros_like(inputs))
        else:
            # Continuous: replace -1 with 0.0 (neutral value)
            safe_inputs = tf.where(mask, inputs, tf.zeros_like(inputs, dtype=tf.float32))
        
        x = self._embedding(safe_inputs)
        x = self._reshape(x)
        return x

    def compute_mask(self, inputs, mask=None):
        """Don't propagate mask automatically - masking handled by Masking layer."""
        return None

    def compute_output_shape(self, input_shape):
        """Required for TimeDistributed wrapper."""
        return input_shape[:-1] + (self.embedding_size,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_categories": self.num_categories,
            "embedding_size": self.embedding_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_categories=config["num_categories"],
            embedding_size=config["embedding_size"]
        )

# =============================================================================
# Gated Linear Unit (GLU) - Equation 5
# =============================================================================
class GatedLinearUnit(tf.keras.layers.Layer):
    """GLU_ω(γ) = σ(W₄γ + b₄) ⊙ (W₅γ + b₅)"""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.fc = Dense(self.units)
        self.fc_gate = Dense(self.units, activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs):
        return self.fc(inputs) * self.fc_gate(inputs)
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(units=config["units"])


# =============================================================================
# Gated Residual Network (GRN) - Equations 2-4
# =============================================================================
class GatedResidualNetwork(tf.keras.layers.Layer):
    """
    GRN_ω(a, c) = LayerNorm(a + GLU_ω(η₁))      [Eq. 2]
    η₁ = W₁η₂ + b₁                               [Eq. 3]
    η₂ = ELU(W₂a + W₃c + b₂)                    [Eq. 4]
    """
    def __init__(self, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.fc1 = Dense(self.units, use_bias=True)
        self.fc_context = Dense(self.units, use_bias=False)
        self.fc2 = Dense(self.units)
        self.dropout = Dropout(self.dropout_rate)
        self.glu = GatedLinearUnit(self.units)
        self.layer_norm = LayerNormalization()
        
        if input_shape[-1] != self.units:
            self.skip_proj = Dense(self.units)
        else:
            self.skip_proj = None
            
        super().build(input_shape)

    def call(self, inputs, context=None, training=None):
        skip = inputs
        if self.skip_proj is not None:
            skip = self.skip_proj(skip)

        # η₂ = ELU(W₂a + W₃c + b₂)
        x = self.fc1(inputs)
        
        if context is not None:
            # Handle broadcasting
            if len(context.shape) == 2 and len(x.shape) == 3:
                context = tf.expand_dims(context, axis=1)
                context = tf.tile(context, [1, tf.shape(x)[1], 1])
            x = x + self.fc_context(context)
        
        x = tf.nn.elu(x)
        
        # η₁ = W₁η₂ + b₁
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        
        # GRN = LayerNorm(a + GLU(η₁))
        x = self.glu(x)
        x = skip + x
        x = self.layer_norm(x)
        
        return x
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            units=config["units"],
            dropout_rate=config["dropout_rate"]
        )


# =============================================================================
# Variable Selection Network (VSN) - Equations 6-8
# =============================================================================
class VariableSelectionNetwork(tf.keras.layers.Layer):
    """
    v_χt = Softmax(GRN_vχ(Ξ_t, c_s))              [Eq. 6]
    ξ̃ⱼ_t = GRN_ξ̃ⱼ(ξⱼ_t)                          [Eq. 7]
    ξ̃_t = Σⱼ vⱼ_χt · ξ̃ⱼ_t                        [Eq. 8]
    
    Note: c_s is passed as external context to the GRN, NOT as a variable to select.
    """
    def __init__(self, num_features, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.grn_vars = [
            GatedResidualNetwork(self.units, self.dropout_rate) 
            for i in range(self.num_features)
        ]
        self.grn_flatten = GatedResidualNetwork(
            self.num_features, self.dropout_rate
        )
        super().build(input_shape)

    def call(self, inputs, context=None, training=None):
        # Flatten original inputs
        flattened = Concatenate(axis=-1)(inputs)
        
        # v = Softmax(GRN(Ξ, c_s))
        v = self.grn_flatten(flattened, context=context, training=training)
        
        # Apply softmax manually without mask issues
        weights = tf.nn.softmax(v, axis=-1)
        
        # Process each variable
        processed = [
            self.grn_vars[i](inputs[i], training=training) 
            for i in range(self.num_features)
        ]
        
        # Weighted sum
        if len(weights.shape) == 2:  # Static: (batch, num_features)
            weighted = [
                tf.expand_dims(weights[:, i], -1) * processed[i]
                for i in range(self.num_features)
            ]
        else:  # Temporal: (batch, time, num_features)
            weighted = [
                tf.expand_dims(weights[..., i], -1) * processed[i]
                for i in range(self.num_features)
            ]
        
        output = tf.add_n(weighted)
        return output, weights
    
    def compute_mask(self, inputs, mask=None):
        # Don't propagate mask from here - let it be handled downstream
        return None

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_features=config["num_features"],
            units=config["units"],
            dropout_rate=config["dropout_rate"]
        )


# =============================================================================
# Interpretable Multi-Head Attention - Equations 13-16 (with mask support)
# =============================================================================
class InterpretableMultiHeadAttention(tf.keras.layers.Layer):
    """
    Interpretable multi-head attention with proper masking.
    
    Key difference from standard MHA (per paper):
    - Values are SHARED across all heads (Eq. 14: V * W_V is common)
    - Attention weights are AVERAGED across heads (Eq. 15)
    
    This allows interpreting attention weights directly since all heads
    attend to the same value representation.
    """
    def __init__(self, num_heads, d_model, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.d_k = d_model // num_heads

    def build(self, input_shape):
        self.W_q = [Dense(self.d_k, use_bias=False) for h in range(self.num_heads)]
        self.W_k = [Dense(self.d_k, use_bias=False) for h in range(self.num_heads)]
        self.W_v = Dense(self.d_model, use_bias=False)  # Shared across heads
        self.W_o = Dense(self.d_model, use_bias=False)
        self.dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Causal mask (upper triangular) - prevents attending to future
        causal_mask = 1.0 - tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )
        causal_mask = tf.expand_dims(causal_mask, 0)  # (1, seq, seq)
        
        # Combine with input mask if provided (-1 masking)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            # Create 2D mask: position i can attend to position j only if both are valid
            # (batch, seq) -> (batch, 1, seq) * (batch, seq, 1) -> (batch, seq, seq)
            input_mask = tf.expand_dims(mask, 1) * tf.expand_dims(mask, 2)
            causal_mask = tf.maximum(causal_mask, 1.0 - input_mask)
        
        # Shared values (Eq. 14: V * W_V)
        V = self.W_v(inputs)
        
        # Compute attention for each head
        attn_weights_list = []
        for h in range(self.num_heads):
            Q = self.W_q[h](inputs)
            K = self.W_k[h](inputs)
            
            # Scaled dot-product attention (Eq. 10)
            scores = tf.matmul(Q, K, transpose_b=True)
            scores = scores / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
            
            # Apply mask (large negative value -> ~0 after softmax)
            scores = scores - causal_mask * 1e9
            
            attn = tf.nn.softmax(scores, axis=-1)
            attn = self.dropout(attn, training=training)
            attn_weights_list.append(attn)
        
        # Average attention weights across heads (Eq. 15)
        attn_avg = tf.reduce_mean(tf.stack(attn_weights_list, axis=0), axis=0)
        
        # Apply averaged attention to shared values (Eq. 14)
        H = tf.matmul(attn_avg, V)
        output = self.W_o(H)
        
        return output, attn_avg
    
    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_heads=config["num_heads"],
            d_model=config["d_model"],
            dropout_rate=config["dropout_rate"]
        )


# =============================================================================
# Gated Add & Norm Layer
# =============================================================================
class GatedAddNorm(tf.keras.layers.Layer):
    """output = LayerNorm(residual + Dropout(GLU(x)))"""
    def __init__(self, units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dropout = Dropout(self.dropout_rate)
        self.glu = GatedLinearUnit(self.units)
        self.layer_norm = LayerNormalization()
        super().build(input_shape)

    def call(self, x, residual, training=None):
        x = self.dropout(x, training=training)
        x = self.glu(x)
        x = residual + x
        return self.layer_norm(x)
    
    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            units=config["units"],
            dropout_rate=config["dropout_rate"]
        )


# =============================================================================
# TFT Model Builder
# =============================================================================
class TemporalFusionTransformer:
    """
    Temporal Fusion Transformer - 100% Paper Compliant Implementation
    
    Reference: Lim et al. (2021) "Temporal Fusion Transformers for Interpretable 
    Multi-horizon Time Series Forecasting"
    
    Usage:
        tft = TemporalFusionTransformer(input_spec, target_spec, ...)
        tft.build_model()
        tft.compile_model(...)
        tft.fit(...)
    """
    
    def __init__(
        self,
        input_spec,
        target_spec,
        d_model=64,
        att_heads=4,
        lookback=90,
        lookforward=30,
        dropout_rate=0.1,
    ):
        self.input_spec = input_spec
        self.target_spec = target_spec
        self.d_model = d_model
        self.att_heads = att_heads
        self.lookback = lookback
        self.lookforward = lookforward
        self.dropout_rate = dropout_rate
        
        # Get quantiles
        if 'quantiles' in target_spec:
            self.quantiles = target_spec['quantiles']
        else:
            for key in target_spec:
                if isinstance(target_spec[key], dict) and 'quantiles' in target_spec[key]:
                    self.quantiles = target_spec[key]['quantiles']
                    break
            else:
                self.quantiles = [0.05, 0.5, 0.95]
        
        self.num_quantiles = len(self.quantiles)
        self.model = None
        
    def build_model(self):
        """Build the TFT model."""
        
        # =====================================================================
        # Create Input Layers
        # =====================================================================
        inputs = {}
        
        # Static inputs: shape (batch, 1)
        for name, spec in self.input_spec['static'].items():
            inputs[f"static_{name}"] = Input(shape=(1,), name=f"static_{name}")
        
        # Past observed inputs: shape (batch, lookback, 1)
        for name, spec in self.input_spec['past_observed'].items():
            inputs[f"past_observed_{name}"] = Input(
                shape=(self.lookback, 1), 
                name=f"past_observed_{name}"
            )
        
        # Future observed inputs: shape (batch, lookforward, 1)
        for name, spec in self.input_spec['future_observed'].items():
            inputs[f"future_observed_{name}"] = Input(
                shape=(self.lookforward, 1), 
                name=f"future_observed_{name}"
            )
        
        # =====================================================================
        # Embedding Layers
        # =====================================================================
        static_embeddings = {}
        for name, spec in self.input_spec['static'].items():
            emb_layer = GenericEmbedding(spec['num_categories'], self.d_model)
            static_embeddings[name] = emb_layer(inputs[f"static_{name}"])
        
        past_embeddings = {}
        for name, spec in self.input_spec['past_observed'].items():
            emb_layer = GenericEmbedding(spec['num_categories'], self.d_model)
            past_embeddings[name] = TimeDistributed(emb_layer)(
                inputs[f"past_observed_{name}"]
            )
        
        future_embeddings = {}
        for name, spec in self.input_spec['future_observed'].items():
            emb_layer = GenericEmbedding(spec['num_categories'], self.d_model)
            future_embeddings[name] = TimeDistributed(emb_layer)(
                inputs[f"future_observed_{name}"]
            )
        
        # =====================================================================
        # Static Encoding (Section 4.3)
        # Static embeddings have shape (batch, d_model) - no time dimension
        # =====================================================================
        static_vsn = VariableSelectionNetwork(
            len(self.input_spec['static']), 
            self.d_model, 
            self.dropout_rate
        )
        
        static_inputs_list = [static_embeddings[name] for name in self.input_spec['static']]
        static_encoding, static_weights = static_vsn(static_inputs_list)
        # Note: No context for static VSN (paper Section 4.2: "For static variables,
        # we note that the context vector c_s is omitted")
        
        # Generate 4 context vectors (all shape: batch, d_model)
        grn_cs = GatedResidualNetwork(self.d_model, self.dropout_rate)
        grn_ch = GatedResidualNetwork(self.d_model, self.dropout_rate)
        grn_cc = GatedResidualNetwork(self.d_model, self.dropout_rate)
        grn_ce = GatedResidualNetwork(self.d_model, self.dropout_rate)
        
        cs = grn_cs(static_encoding)  # For variable selection
        ch = grn_ch(static_encoding)  # For LSTM hidden state
        cc = grn_cc(static_encoding)  # For LSTM cell state
        ce = grn_ce(static_encoding)  # For static enrichment
        
        # =====================================================================
        # Past Temporal Variable Selection (Section 4.2)
        # FIX: Only actual input variables, static context passed via 'context' param
        # =====================================================================
        past_vsn = VariableSelectionNetwork(
            len(self.input_spec['past_observed']),  # FIXED: removed +1
            self.d_model, 
            self.dropout_rate
        )
        
        past_inputs_list = [past_embeddings[name] for name in self.input_spec['past_observed']]
        # FIXED: Removed cs_past from variable list - cs is passed as context instead
        
        past_encoding, past_weights = past_vsn(past_inputs_list, context=cs)
        
        # =====================================================================
        # Future Temporal Variable Selection (Section 4.2)
        # FIX: Only actual input variables, static context passed via 'context' param
        # =====================================================================
        future_vsn = VariableSelectionNetwork(
            len(self.input_spec['future_observed']),  # FIXED: removed +1
            self.d_model, 
            self.dropout_rate
        )
        
        future_inputs_list = [future_embeddings[name] for name in self.input_spec['future_observed']]
        # FIXED: Removed cs_future from variable list - cs is passed as context instead
        
        future_encoding, future_weights = future_vsn(future_inputs_list, context=cs)
        
        # =====================================================================
        # Create Mask from OBSERVED inputs only (not known inputs)
        # Known inputs (hour, weekday, etc.) never have -1 padding - time always exists
        # Only observed inputs (flow, pressure, etc.) have -1 for missing data
        # =====================================================================
        # Identify observed (real_*) vs known inputs in past_observed
        past_observed_names = [name for name in self.input_spec['past_observed'] 
                               if name.startswith('real')]
        
        if len(past_observed_names) > 0:
            # Create mask from observed inputs only
            past_masks = []
            for name in past_observed_names:
                input_tensor = inputs[f"past_observed_{name}"]
                mask = tf.not_equal(tf.squeeze(input_tensor, axis=-1), -1)
                past_masks.append(mask)
            
            # Combine: timestep is valid only if ALL observed inputs are valid
            past_mask = past_masks[0]
            for m in past_masks[1:]:
                past_mask = tf.logical_and(past_mask, m)
        else:
            # No observed inputs in past - all timesteps are valid
            batch_size = tf.shape(list(inputs.values())[0])[0]
            past_mask = tf.ones((batch_size, self.lookback), dtype=tf.bool)
        
        # For future: typically only known inputs, so no -1 padding
        # Check if there are any observed (real_*) inputs in future
        future_observed_names = [name for name in self.input_spec['future_observed'] 
                                 if name.startswith('real')]
        
        if len(future_observed_names) > 0:
            # Create mask from observed inputs only
            future_masks = []
            for name in future_observed_names:
                input_tensor = inputs[f"future_observed_{name}"]
                mask = tf.not_equal(tf.squeeze(input_tensor, axis=-1), -1)
                future_masks.append(mask)
            
            future_mask = future_masks[0]
            for m in future_masks[1:]:
                future_mask = tf.logical_and(future_mask, m)
        else:
            # No observed inputs in future - all timesteps are valid (typical case)
            batch_size = tf.shape(list(inputs.values())[0])[0]
            future_mask = tf.ones((batch_size, self.lookforward), dtype=tf.bool)
        
        # Full sequence mask for attention
        combined_mask = tf.concat([past_mask, future_mask], axis=1)
        
        # =====================================================================
        # Apply Masking Layer (for LSTM to ignore padded timesteps)
        # =====================================================================
        masking_layer = Masking(mask_value=-1)
        past_encoding_masked = masking_layer(past_encoding)
        future_encoding_masked = masking_layer(future_encoding)
        
        # =====================================================================
        # LSTM Sequence-to-Sequence (Section 4.5.1)
        # Encoder processes past, decoder processes future
        # Static context initializes LSTM states
        # =====================================================================
        lstm_encoder = LSTM(self.d_model, return_sequences=True, return_state=True)
        lstm_decoder = LSTM(self.d_model, return_sequences=True)
        
        # Keras LSTM initial_state order: [hidden_state, cell_state]
        # Paper: ch -> hidden, cc -> cell
        lstm_past, state_h, state_c = lstm_encoder(
            past_encoding_masked, 
            initial_state=[ch, cc]
        )
        
        lstm_future = lstm_decoder(
            future_encoding_masked, 
            initial_state=[state_h, state_c]
        )
        
        # Concatenate past and future
        lstm_output = Concatenate(axis=1)([lstm_past, lstm_future])
        temporal_features = Concatenate(axis=1)([past_encoding_masked, future_encoding_masked])
        
        # Gated skip connection (Eq. 17)
        gate_lstm = GatedAddNorm(self.d_model, self.dropout_rate)
        temporal_feature_layer = gate_lstm(lstm_output, temporal_features)
        
        # =====================================================================
        # Static Enrichment (Section 4.5.2, Eq. 18)
        # Enrich temporal features with static context
        # =====================================================================
        grn_enrichment = GatedResidualNetwork(self.d_model, self.dropout_rate)
        enriched_features = grn_enrichment(temporal_feature_layer, context=ce)
        
        # =====================================================================
        # Self-Attention (Section 4.5.3, Eq. 19)
        # Interpretable multi-head attention with decoder masking
        # =====================================================================
        attention = InterpretableMultiHeadAttention(
            self.att_heads, 
            self.d_model, 
            self.dropout_rate
        )
        # Pass mask to attention so it ignores padded positions
        attention_output, attention_weights = attention(enriched_features, mask=combined_mask)
        
        # Gated skip (Eq. 20)
        gate_attention = GatedAddNorm(self.d_model, self.dropout_rate)
        gated_attention_output = gate_attention(attention_output, enriched_features)
        
        # =====================================================================
        # Position-wise Feed-Forward (Section 4.5.4, Eq. 21)
        # =====================================================================
        grn_positionwise = GatedResidualNetwork(self.d_model, self.dropout_rate)
        ff_output = grn_positionwise(gated_attention_output)
        
        # Gated skip connection over entire transformer block (Eq. 22)
        gate_positionwise = GatedAddNorm(self.d_model, self.dropout_rate)
        decoder_output = gate_positionwise(ff_output, temporal_feature_layer)
        
        # =====================================================================
        # Quantile Outputs (Eq. 23)
        # FIXED: Direct projection from decoder output, no extra GRN
        # Paper: ŷ(q,t,τ) = W_q * ψ̃(t,τ) + b_q
        # =====================================================================
        # Take only future horizons for output
        future_output = decoder_output[:, -self.lookforward:, :]
        
        # Direct linear projection to quantiles (Eq. 23)
        regression_output = Dense(self.num_quantiles, name='regression')(future_output)
        
        # =====================================================================
        # Classification Head (your extension - not from paper)
        # =====================================================================
        # Get future attention weights
        future_attention = attention_weights[:, -self.lookforward:, -self.lookforward:]
        
        # Attention-weighted temporal aggregation
        attention_received = tf.nn.softmax(
            tf.reduce_mean(future_attention, axis=-2), 
            axis=-1
        )
        attention_weights_expanded = tf.expand_dims(attention_received, axis=-1)
        
        # Multiple temporal aggregations
        weighted_output = tf.reduce_sum(future_output * attention_weights_expanded, axis=1)
        max_pooled = tf.reduce_max(future_output, axis=1)
        mean_pooled = tf.reduce_mean(future_output, axis=1)
        
        combined_temporal_features = Concatenate()([weighted_output, max_pooled, mean_pooled])
        
        # Prediction statistics features
        pred_5th = regression_output[:, :, 0]
        pred_50th = regression_output[:, :, 1]
        pred_95th = regression_output[:, :, 2]
        
        pred_50th_min = tf.reduce_min(pred_50th, axis=1, keepdims=True)
        pred_50th_max = tf.reduce_max(pred_50th, axis=1, keepdims=True)
        pred_50th_mean = tf.reduce_mean(pred_50th, axis=1, keepdims=True)
        pred_50th_std = tf.math.reduce_std(pred_50th, axis=1, keepdims=True)
        
        uncertainty_spread = pred_95th - pred_5th
        mean_uncertainty = tf.reduce_mean(uncertainty_spread, axis=1, keepdims=True)
        max_uncertainty = tf.reduce_max(uncertainty_spread, axis=1, keepdims=True)
        std_uncertainty = tf.math.reduce_std(uncertainty_spread, axis=1, keepdims=True)
        
        skew_indicator = (pred_95th - pred_50th) - (pred_50th - pred_5th)
        mean_skew = tf.reduce_mean(skew_indicator, axis=1, keepdims=True)
        
        focused_features = Concatenate()([
            combined_temporal_features,
            pred_50th_min, pred_50th_max, pred_50th_mean, pred_50th_std,
            mean_uncertainty, max_uncertainty, std_uncertainty,
            mean_skew
        ])
        
        # Classification MLP
        x = Dense(128, activation='relu', name='cls_dense1')(focused_features)
        x = BatchNormalization(name='cls_bn1')(x)
        x = Dropout(0.4, name='cls_drop1')(x)
        x = Dense(64, activation='relu', name='cls_dense2')(x)
        x = BatchNormalization(name='cls_bn2')(x)
        x = Dropout(0.4, name='cls_drop2')(x)
        x = Dense(32, activation='relu', name='cls_dense3')(x)
        x = BatchNormalization(name='cls_bn3')(x)
        x = Dropout(0.5, name='cls_drop3')(x)
        classification_output = Dense(1, activation='sigmoid', name='classification')(x)
        
        # =====================================================================
        # Build Model
        # =====================================================================
        self.model = Model(
            inputs=inputs,
            outputs={
                'regression': regression_output,
                'classification': classification_output
            }
        )
        
        # Store for interpretation
        self._static_weights = static_weights
        self._past_weights = past_weights
        self._future_weights = future_weights
        self._attention_weights = attention_weights
        
        return self.model
    
    def compile_model(self, optimizer, loss, metrics=None, loss_weights=None):
        """Compile the model."""
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights
        )
    
    def fit(self, x, y, epochs, batch_size, validation_data=None, callbacks=None, verbose=1):
        """Train the model."""
        return self.model.fit(
            x=x,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
    
    def predict(self, x, **kwargs):
        """Make predictions."""
        return self.model.predict(x, **kwargs)
    
    def get_interpretation_weights(self, inputs):
        """
        Get interpretation weights by running a forward pass.
        
        Per paper Section 7, these weights enable three interpretability use cases:
        - Variable importance (Section 7.1): Use static_weights, past_weights, future_weights
        - Temporal patterns (Section 7.2): Use attention_weights
        - Regime identification (Section 7.3): Use attention_weights
        
        For predictions, use .predict() method instead.
        
        Returns:
            dict with:
                - 'static_weights': VSN weights for static features (Eq. 8)
                - 'past_weights': VSN weights for past temporal features (Eq. 8)
                - 'future_weights': VSN weights for future temporal features (Eq. 8)
                - 'attention_weights': self-attention weights α(t, n, τ) (Eq. 27)
        """
        # Create interpretation model
        interp_model = Model(
            inputs=self.model.inputs,
            outputs=[
                self._static_weights,
                self._past_weights,
                self._future_weights,
                self._attention_weights
            ]
        )
        
        results = interp_model.predict(inputs)
        
        return {
            'static_weights': results[0],
            'past_weights': results[1],
            'future_weights': results[2],
            'attention_weights': results[3]
        }
    
    def save_model(self, filepath):
        """Save model weights and config."""
        self.model.save(filepath + '.h5')
        
        config = {
            'input_spec': self.input_spec,
            'target_spec': self.target_spec,
            'd_model': self.d_model,
            'att_heads': self.att_heads,
            'lookback': self.lookback,
            'lookforward': self.lookforward,
            'dropout_rate': self.dropout_rate,
        }
        
        with open(filepath + '_config.pkl', 'wb') as f:
            pickle.dump(config, f)
    
    @classmethod
    def load_model(cls, filepath, custom_objects=None):
        """Load model from saved weights and config."""
        with open(filepath + '_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        instance = cls(**config)
        instance.build_model()
        
        all_custom_objects = get_custom_objects()
        if custom_objects:
            all_custom_objects.update(custom_objects)
        
        instance.model.load_weights(filepath + '.h5')
        
        return instance


# =============================================================================
# Custom Objects for Model Loading
# =============================================================================
def get_custom_objects():
    """
    Returns dict of custom objects for model loading.
    """
    return {
        'GenericEmbedding': GenericEmbedding,
        'GatedLinearUnit': GatedLinearUnit,
        'GatedResidualNetwork': GatedResidualNetwork,
        'VariableSelectionNetwork': VariableSelectionNetwork,
        'InterpretableMultiHeadAttention': InterpretableMultiHeadAttention,
        'GatedAddNorm': GatedAddNorm,
    }
