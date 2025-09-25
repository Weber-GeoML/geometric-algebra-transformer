# GATR Repository Overview

A comprehensive guide to the Geometric Algebra Transformer codebase - estimated reading time: 8-10 minutes.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Modules Deep Dive](#core-modules-deep-dive)
3. [Data Flow and Representations](#data-flow-and-representations)
4. [Key Components](#key-components)
5. [Experiments and Usage](#experiments-and-usage)
6. [Testing and Utilities](#testing-and-utilities)

## Architecture Overview

GATR is a transformer architecture that operates on geometric data using projective geometric algebra (PGA). The key innovation is representing geometric objects (points, planes, rotations, etc.) as 16-dimensional multivectors in G(3,0,1) and maintaining Pin(3,0,1) equivariance throughout the network.

### High-Level Architecture
```
Input Geometry → Multivector Embedding → GATr Blocks → Output Projection → Decoded Geometry
     ↓                    ↓                    ↓              ↓                ↓
   Points,            (..., channels, 16)   Attention +    (..., out_ch, 16)  Points,
   Planes,               +                    MLP +                            Planes,
   etc.              (..., s_channels)    Residuals                          etc.
```

## Core Modules Deep Dive

### 1. `/gatr/interface/` - Geometric Embeddings
**Purpose**: Converts real-world geometric objects to/from multivector representations.

**Key Files**:
- `point.py`: 3D points → trivectors (components 11-14)
- `plane.py`: Oriented planes → vectors (components 2-4) 
- `rotation.py`: Quaternions → bivectors (components 0, 8-10)
- `translation.py`: 3D translations → motors
- `object.py`: Complete 3D objects (position + orientation)
- `scalar.py`, `pseudoscalar.py`: Scalar/pseudoscalar embeddings

**Data Flow**: 
```python
# Example: Point embedding
coordinates = torch.tensor([1.0, 2.0, 3.0])  # (3,)
multivector = embed_point(coordinates)        # (16,) - embedded as trivector
```

### 2. `/gatr/primitives/` - Core GA Operations
**Purpose**: Fundamental geometric algebra operations that preserve equivariance.

**Key Files**:
- `linear.py`: **EquiLinear operations** - the core equivariant linear maps using 9 basis elements
- `bilinear.py`: Geometric product, outer product operations
- `attention.py`: Geometric attention mechanisms with PGA inner products
- `invariants.py`: Inner products, norms, and other invariant functions
- `dual.py`: Dual operations and equivariant join
- `nonlinearities.py`: Gated nonlinear activations
- `normalization.py`: Geometric layer normalization

**Critical Component - EquiLinear**:
```python
def equi_linear(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant linear map using 9 precomputed basis elements"""
    # x: (..., in_channels, 16)
    # coeffs: (out_channels, in_channels, 9) 
    # Returns: (..., out_channels, 16)
```

### 3. `/gatr/layers/` - Network Layers
**Purpose**: Stateful nn.Module implementations of GA operations.

#### `/gatr/layers/linear.py` - **EquiLinear Layer**
- **Most important layer** - handles MV→MV and Scalar→Scalar transformations
- Interface: `forward(multivectors, scalars=None) → (output_mv, output_s)`
- Supports 4 initialization schemes: "default", "small", "unit_scalar", "almost_unit_scalar"
- Used everywhere: input/output projections, attention Q/K/V, MLP layers

#### `/gatr/layers/attention/`
- `self_attention.py`: Main geometric self-attention layer
- `qkv.py`: Query/Key/Value projection modules (uses EquiLinear)
- `attention.py`: Core geometric attention computation
- `cross_attention.py`: Cross-attention variant
- `config.py`: Configuration dataclass

#### `/gatr/layers/mlp/`
- `mlp.py`: **GeoMLP** - geometric MLP using bilinears + EquiLinear
- `geometric_bilinears.py`: GP + equivariant join operations
- `nonlinearities.py`: Scalar-gated nonlinearities
- `config.py`: MLP configuration

#### Other Layers:
- `gatr_block.py`: **Main transformer block** - combines attention + MLP + residuals
- `layer_norm.py`: Geometric layer normalization
- `dropout.py`: Grade-aware dropout

### 4. `/gatr/nets/` - Complete Architectures
**Purpose**: Full network architectures combining all components.

**Key Files**:
- `gatr.py`: **Main GATr architecture** for single token dimension
- `axial_gatr.py`: Axial attention variant for 2D token grids

**GATr Architecture**:
```python
class GATr(nn.Module):
    def __init__(self, in_mv_channels, out_mv_channels, hidden_mv_channels, 
                 in_s_channels, out_s_channels, hidden_s_channels,
                 attention, mlp, num_blocks=10):
        
        self.linear_in = EquiLinear(...)      # Input projection
        self.blocks = nn.ModuleList([         # Transformer blocks
            GATrBlock(attention=attention, mlp=mlp) 
            for _ in range(num_blocks)
        ])
        self.linear_out = EquiLinear(...)     # Output projection
```

### 5. `/gatr/experiments/` - Experiment Management
**Purpose**: Task-specific wrappers and experiment orchestration.

#### `/gatr/experiments/nbody/`
- `experiment.py`: N-body physics experiment manager
- `dataset.py`: N-body data loading and preprocessing
- `wrappers.py`: Task-specific GATr wrappers
- `simulator.py`: Physics simulation

#### `/gatr/experiments/arteries/`
- `experiment.py`: Arterial wall-shear-stress experiment
- `dataset.py`: Medical data loading
- `wrappers.py`: Mesh-based GATr wrappers

**Pattern**: Each experiment has its own wrapper that:
1. Embeds task-specific data into multivectors
2. Configures GATr for the task
3. Extracts task-specific outputs
4. Handles training/evaluation logic

### 6. `/gatr/baselines/` - Comparison Models
**Purpose**: Non-geometric baselines for comparison.

**Key Files**:
- `mlp.py`: **Simple MLP baseline** - flattens inputs, standard MLP
- `transformer.py`: Standard transformer (no geometric structure)
- `gcan.py`: GCA-MLP and GCA-GNN baselines
- `segnn.py`: SEGNN baseline

### 7. `/gatr/utils/` - Utilities
**Purpose**: Supporting functionality for the main library.

**Key Files**:
- `einsum.py`: Optimized Einstein summation with caching
- `compile_linear.py`: Fast inference via compiled EquiLinear layers
- `tensors.py`: Tensor manipulation utilities
- `mlflow.py`: MLflow logging integration
- `clifford.py`: Non-differentiable GA operations

## Data Flow and Representations

### Multivector Structure (16 components)
Following the `clifford` library convention:
```
[scalar, e0, e1, e2, e3, e01, e02, e03, e12, e13, e23, e012, e013, e023, e123, e0123]
 [  0  ] [1-4] [     5-10     ] [    11-14    ] [15]
 scalar  vector   bivector       trivector    pseudoscalar
```

### Data Shapes Throughout Network
- **Input**: `(..., items, in_mv_channels, 16)` + optional `(..., items, in_s_channels)`
- **Hidden**: `(..., items, hidden_mv_channels, 16)` + optional `(..., items, hidden_s_channels)`
- **Output**: `(..., items, out_mv_channels, 16)` + optional `(..., items, out_s_channels)`

### Typical Data Flow
1. **Embedding**: Real geometry → Multivectors via `/interface/`
2. **Input Projection**: `EquiLinear` maps to hidden dimensions
3. **Processing**: N × `GATrBlock` (attention + MLP + residuals)
4. **Output Projection**: `EquiLinear` maps to output dimensions
5. **Extraction**: Multivectors → Real geometry via `/interface/`

## Key Components

### EquiLinear Layer (Central to Everything)
**Location**: `/gatr/layers/linear.py`
**Usage**: Input/output projections, attention Q/K/V, MLP layers, geometric bilinears

**Interface**:
```python
def forward(self, multivectors, scalars=None):
    # multivectors: (..., in_mv_channels, 16)
    # scalars: (..., in_s_channels) or None
    # Returns: (output_mv, output_s)
```

**Used In**:
- `GATr.linear_in` and `GATr.linear_out`
- `SelfAttention.out_linear`
- `QKVModule.in_linear`
- `GeometricBilinear.linear_*`
- `GeoMLP` internal layers

### GATrBlock (Main Processing Unit)
**Location**: `/gatr/layers/gatr_block.py`
**Structure**:
```python
def forward(self, multivectors, scalars, reference_mv=None, ...):
    # Attention block
    h_mv, h_s = self.norm(multivectors, scalars)
    h_mv, h_s = self.attention(h_mv, h_s, ...)
    multivectors = multivectors + h_mv  # Residual
    scalars = scalars + h_s
    
    # MLP block  
    h_mv, h_s = self.norm(multivectors, scalars)
    h_mv, h_s = self.mlp(h_mv, h_s, reference_mv)
    multivectors = multivectors + h_mv  # Residual
    scalars = scalars + h_s
    
    return multivectors, scalars
```

### Geometric Attention
**Location**: `/gatr/primitives/attention.py`
**Key Innovation**: Uses PGA inner products + nonlinear distance features
```python
# Attention weights combine:
# 1. PGA inner product between multivectors
# 2. Euclidean inner product between scalars  
# 3. Nonlinear distance-aware features
```

## Experiments and Usage

### N-body Physics
- **Data**: Point masses with positions/velocities
- **Task**: Predict future trajectories
- **Embedding**: Points as trivectors, velocities as vectors
- **Architecture**: Standard GATr with ~10 blocks

### Arterial Wall-Shear-Stress
- **Data**: 3D mesh surfaces with flow data
- **Task**: Predict wall shear stress from geometry
- **Embedding**: Mesh vertices as points, faces as planes
- **Architecture**: GATr with mesh-specific attention masking

## Testing and Utilities

### `/tests/` - Comprehensive Testing
- **Equivariance tests**: Verify Pin(3,0,1) equivariance for all layers
- **Unit tests**: Individual component functionality
- **Integration tests**: End-to-end network behavior
- **Regression tests**: Training simple tasks to completion

### Key Testing Pattern
```python
def test_equivariance():
    # Apply random Pin(3,0,1) transformation
    transformed_input = transform(input)
    
    # Check: f(T(x)) = T(f(x))
    assert torch.allclose(
        layer(transformed_input),
        transform(layer(input))
    )
```

## Summary

The GATR codebase is built around the central concept of **EquiLinear layers** that maintain geometric equivariance while processing multivector representations of geometric data. The architecture follows a clear hierarchy:

1. **Primitives** provide the mathematical foundation
2. **Layers** wrap primitives in stateful modules  
3. **Nets** combine layers into complete architectures
4. **Interface** handles real-world geometry conversion
5. **Experiments** provide task-specific implementations

The key insight is that by representing all geometric objects in a unified 16-dimensional multivector space and using equivariant operations throughout, GATR can process diverse geometric data while maintaining mathematical guarantees about geometric transformations.