# Architecture

> **Relevant source files**
> * [README.md](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/README.md)
> * [model.py](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py)
> * [trainer.py](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py)

This document provides a comprehensive overview of the system architecture of the PointNetLK_Revisited repository. It describes the high-level system design, component interactions, and data flow during both training and testing phases. For detailed information about the algorithm itself, see [PointNetLK Algorithm](/Lilac-Lee/PointNetLK_Revisited/3.1-pointnetlk-algorithm).

## System Components and Interactions

The PointNetLK_Revisited system is designed with a clean separation of concerns, with specialized components for model definition, training logic, and data processing.

```

```

Sources: [trainer.py L19-L43](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py#L19-L43)

 [model.py L50-L101](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L50-L101)

 [model.py L103-L351](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L103-L351)

### Key Classes and Their Responsibilities

The core architecture consists of several key classes that work together to implement the PointNetLK algorithm:

| Class | File | Primary Responsibility |
| --- | --- | --- |
| `TrainerAnalyticalPointNetLK` | trainer.py | Orchestrates the training process, creates models, and evaluates performance |
| `Pointnet_Features` | model.py | Implements the PointNet feature extraction network |
| `AnalyticalPointNetLK` | model.py | Implements the Lucas-Kanade algorithm for point cloud registration |
| `MLPNet` | model.py | Multi-layer perceptron used in the feature extraction network |

Sources: [trainer.py L19-L43](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py#L19-L43)

 [model.py L29-L41](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L29-L41)

 [model.py L50-L101](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L50-L101)

 [model.py L103-L351](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L103-L351)

## Model Architecture

The core algorithm combines PointNet for feature extraction with Lucas-Kanade for iterative pose refinement.

```mermaid
classDiagram
    class Pointnet_Features {
        +mlp1: MLPNet(3→64)
        +mlp2: MLPNet(64→128)
        +mlp3: MLPNet(128→dim_k)
        +forward(points, iter)
    }
    class AnalyticalPointNetLK {
        +ptnet: Pointnet_Features
        +device
        +g: Transformation matrix
        +forward(p0, p1, mode, maxiter, xtol)
        +iclk_new(g0, p0, p1, maxiter, xtol, mode)
        +Cal_Jac(Mask_fn, A_fn, Ax_fn, BN_fn, max_idx)
        +update(g, dx)
    }
    class MLPNet {
        +layers: Sequential
        +forward(inp)
    }
    class TrainerAnalyticalPointNetLK {
        +create_features()
        +create_model()
        +train_one_epoch()
        +eval_one_epoch()
        +test_one_epoch()
        +compute_loss()
    }
    Pointnet --> Features : "uses"
    AnalyticalPointNetLK --> Pointnet : "uses"
    TrainerAnalyticalPointNetLK --> Pointnet : "creates"
    TrainerAnalyticalPointNetLK --> AnalyticalPointNetLK : "creates"
```

Sources: [model.py L50-L101](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L50-L101)

 [model.py L103-L351](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L103-L351)

 [trainer.py L19-L43](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py#L19-L43)

### Feature Extraction Architecture

The feature extraction component is built on the PointNet architecture with a series of MLP layers:

```mermaid
flowchart TD

subgraph Pointnet_Features_Pipeline ["Pointnet_Features Pipeline"]
end

input["Input Points [B,N,3]"]
transpose["Transpose [B,3,N]"]
mlp1["MLPNet(3→64)"]
mlp2["MLPNet(64→128)"]
mlp3["MLPNet(128→dim_k)"]
maxpool["MaxPool1d"]
output["Output Features [B,dim_k]"]

    input --> transpose
    transpose --> mlp1
    mlp1 --> mlp2
    mlp2 --> mlp3
    mlp3 --> maxpool
    maxpool --> output
```

Sources: [model.py L50-L101](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L50-L101)

### Lucas-Kanade Algorithm Implementation

The AnalyticalPointNetLK class implements the iterative closest point registration with Lucas-Kanade optimization:

```mermaid
flowchart TD

subgraph iterative_closest_point_LK ["iterative_closest_point_LK"]
end

input["Source(p1) & Target(p0)"]
init["Initialize g0"]
feature["Extract features f0, f1"]
jacobian["Calculate Jacobian J"]
pinverse["Compute Pseudo-inverse"]
transform["Transform points"]
residual["Compute residual r = f - f0"]
update["Update transformation g"]
check["Check convergence"]
output["Final transformation g"]

    input --> init
    init --> feature
    feature --> jacobian
    jacobian --> pinverse
    pinverse --> transform
    transform --> residual
    residual --> update
    update --> check
    check --> transform
    check --> output
```

Sources: [model.py L264-L350](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L264-L350)

## Data Flow

The system processes point cloud data through several stages during both training and testing.

```mermaid
flowchart TD

subgraph Evaluation ["Evaluation"]
end

subgraph Registration ["Registration"]
end

subgraph Feature_Extraction ["Feature Extraction"]
end

subgraph Input_Processing ["Input Processing"]
end

p0["Source Point Cloud"]
p1["Target Point Cloud"]
preprocessing["Zero-mean Centering"]
pointnet["PointNet Feature Network"]
features["Global Features"]
lk["Lucas-Kanade Algorithm"]
jacobian["Jacobian Computation"]
optimization["Iterative Optimization"]
transformation["Final Transformation"]
metrics["Compute Error Metrics"]
results["Registration Results"]

    p0 --> preprocessing
    p1 --> preprocessing
    preprocessing --> pointnet
    pointnet --> features
    features --> lk
    lk --> jacobian
    jacobian --> optimization
    optimization --> transformation
    transformation --> metrics
    metrics --> results
```

Sources: [model.py L143-L202](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L143-L202)

 [model.py L264-L350](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/model.py#L264-L350)

 [trainer.py L210-L241](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py#L210-L241)

### Data Flow During Training

During training, the system:

1. Loads batches of point cloud pairs and ground truth transformations
2. Processes the point clouds (zero-mean centering)
3. Extracts features using the PointNet feature extractor
4. Computes the analytical Jacobian
5. Iteratively refines the transformation using the Lucas-Kanade algorithm
6. Computes the loss between estimated and ground truth transformations
7. Updates the model parameters through backpropagation

Sources: [trainer.py L45-L67](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py#L45-L67)

 [trainer.py L210-L241](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py#L210-L241)

### Data Flow During Testing

During testing, the system:

1. Loads test point cloud pairs
2. Optionally applies voxelization for real-world datasets
3. Processes the point clouds (zero-mean centering)
4. Extracts features and computes the analytical Jacobian
5. Iteratively refines the transformation until convergence
6. Computes evaluation metrics (rotation and translation errors)

Sources: [trainer.py L87-L207](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py#L87-L207)

## Integration with Datasets

The system is designed to work with multiple datasets through a unified interface:

```mermaid
flowchart TD

subgraph Processing_Options ["Processing Options"]
end

subgraph Dataset_Loaders ["Dataset Loaders"]
end

modelnet["ModelNet40"]
shapenet["ShapeNet"]
kitti["KITTI"]
threedmatch["3DMatch"]
synthetic["Synthetic Data"]
real["Real-world Data"]
voxel["Voxelization"]
normal["Normal Processing"]
traintest["Training/Testing Pipeline"]

    modelnet --> synthetic
    shapenet --> synthetic
    kitti --> real
    threedmatch --> real
    synthetic --> normal
    real --> voxel
    normal --> traintest
    voxel --> traintest
```

Sources: [trainer.py L94-L107](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py#L94-L107)

 [trainer.py L210-L226](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/trainer.py#L210-L226)

## Conclusion

The PointNetLK_Revisited architecture combines the feature extraction capabilities of PointNet with the iterative refinement approach of the Lucas-Kanade algorithm. The system is designed with clear separation of concerns, allowing for flexibility in data handling and model configuration. The analytical computation of the Jacobian is a key innovation that enables more efficient and accurate point cloud registration compared to previous approaches.

For more details on the specific components, refer to [Core Components](/Lilac-Lee/PointNetLK_Revisited/3-core-components) and [Data Processing](/Lilac-Lee/PointNetLK_Revisited/4-data-processing).