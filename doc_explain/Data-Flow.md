# Data Flow

> **Relevant source files**
> * [README.md](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/README.md)
> * [data_utils.py](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py)
> * [test.py](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/test.py)
> * [train.py](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/train.py)

This document details how data flows through the PointNetLK_Revisited system during both training and testing/inference processes. It covers data loading, preprocessing, model input/output flow, and transformation estimation. For information about the system's architectural components, see [System Design](/Lilac-Lee/PointNetLK_Revisited/2.1-system-design).

## 1. Overview of Data Flow

The PointNetLK_Revisited system handles point cloud registration through a pipeline that transforms raw point cloud data into estimated rigid transformations. The pipeline includes data loading, preprocessing, feature extraction, and iterative transformation estimation.

```mermaid
flowchart TD

subgraph Output ["Output"]
end

subgraph Model_Processing ["Model Processing"]
end

subgraph Preprocessing ["Preprocessing"]
end

subgraph Input ["Input"]
end

Raw["Raw Point Cloud Data"]
Load["Data Loading"]
Norm["Normalization"]
Vox["Voxelization"]
Feat["Feature Extraction"]
LK["Lucas-Kanade Iteration"]
Trans["Transformation Parameters"]

    Raw --> Load
    Load --> Norm
    Norm --> Vox
    Vox --> Feat
    Feat --> LK
    LK --> Trans
```

Sources: [data_utils.py L55-L179](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py#L55-L179)

 [train.py L154-L176](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/train.py#L154-L176)

 [test.py L115-L144](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/test.py#L115-L144)

## 2. Data Loading Process

The system handles different types of datasets with specialized data loading pipelines.

### 2.1 Dataset Loading

```mermaid
flowchart TD

subgraph Output_Format ["Output Format"]
end

subgraph Data_Loaders ["Data Loaders"]
end

subgraph Dataset_Types ["Dataset Types"]
end

modelnet["ModelNet40"]
shapenet["ShapeNet"]
threeDmatch["3DMatch"]
kitti["KITTI"]
mesh2points["Mesh2Points"]
onUnitCube["OnUnitCube"]
resampler["Resampler"]
registration["PointRegistration"]
p0["Source Points (p0)"]
p1["Target Points (p1)"]
igt["Ground Truth Transform (igt)"]
threeDMatchTest["ThreeDMatch_Testing"]
kittiLoader["KITTI Loader"]

    modelnet --> mesh2points
    shapenet --> mesh2points
    mesh2points --> onUnitCube
    onUnitCube --> resampler
    resampler --> registration
    threeDmatch --> threeDMatchTest
    kitti --> kittiLoader
    registration --> p0
    registration --> p1
    registration --> igt
    threeDMatchTest --> p0
    threeDMatchTest --> p1
    threeDMatchTest --> igt
```

Sources: [train.py L154-L176](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/train.py#L154-L176)

 [test.py L115-L144](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/test.py#L115-L144)

 [data_utils.py L294-L344](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py#L294-L344)

### 2.2 Data Transformation Pipeline

During training and testing, point clouds undergo different transformations depending on the dataset type:

| Dataset Type | Preprocessing Steps | Output Format |
| --- | --- | --- |
| ModelNet40/ShapeNet | 1. Mesh to Points2. Normalize to unit cube3. Resample to fixed point count4. Apply random transformation | Source points, target points, ground truth transform |
| 3DMatch | 1. Load point pairs2. Find overlapping regions3. Voxelize point clouds4. Apply specified transformation | Voxelized source points, voxelized target points, ground truth transform |

For synthetic datasets (ModelNet/ShapeNet), the transformation sequence is:

```mermaid
sequenceDiagram
  participant Raw
  participant Points
  participant Normalized
  participant Resampled
  participant Transformed

  Raw->Points: Mesh2Points()
  Points->Normalized: OnUnitCube()
  Normalized->Resampled: Resampler(num_points)
  Resampled->Transformed: RandomTransformSE3(mag)
```

Sources: [data_utils.py L777-L846](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py#L777-L846)

 [data_utils.py L249-L283](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py#L249-L283)

### 2.3 Voxelization Process

For real-world datasets (3DMatch/KITTI), the voxelization process is critical:

```mermaid
flowchart TD

subgraph Voxelization ["Voxelization"]
end

p0["Source Point Cloud (p0)"]
p1["Target Point Cloud (p1)"]
find["find_voxel_overlaps()"]
voxel0["points_to_voxel_second() for p0"]
voxel1["points_to_voxel_second() for p1"]
intersect["Find Intersecting Voxels"]
voxels_p0["Voxelized Source Points"]
voxels_p1["Voxelized Target Points"]
coords_p0["Source Voxel Coordinates"]
coords_p1["Target Voxel Coordinates"]

    p0 --> find
    p1 --> find
    find --> voxel0
    find --> voxel1
    voxel0 --> intersect
    voxel1 --> intersect
```

Sources: [data_utils.py L36-L52](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py#L36-L52)

 [data_utils.py L398-L449](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py#L398-L449)

## 3. Training Data Flow

During training, data flows through multiple stages from loading to loss computation:

```mermaid
flowchart TD

subgraph Optimization ["Optimization"]
end

subgraph Forward_Pass ["Forward Pass"]
end

subgraph Data_Preparation ["Data Preparation"]
end

load["get_datasets()"]
dataloader["DataLoader (batch, shuffle)"]
model["AnalyticalPointNetLK"]
features["Pointnet_Features"]
lk["Lucas-Kanade Algorithm"]
loss["compute_loss()"]
backprop["Backpropagation"]
update["Optimizer Update"]

    load --> dataloader
    dataloader --> model
    model --> features
    features --> lk
    lk --> loss
    loss --> backprop
    backprop --> update
    update --> dataloader
```

Sources: [train.py L84-L146](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/train.py#L84-L146)

 [train.py L128-L131](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/train.py#L128-L131)

The data batches are processed as follows:

1. Source and target point clouds are loaded in batches
2. Both point clouds are passed through PointNet feature extraction
3. Features are used in the Lucas-Kanade algorithm to estimate transformation
4. Loss is computed between estimated and ground truth transformation
5. Gradients are backpropagated to update the model

## 4. Testing/Inference Data Flow

During testing, the data flow is similar but focused on evaluation:

```mermaid
flowchart TD

subgraph Evaluation ["Evaluation"]
end

subgraph Model_Inference ["Model Inference"]
end

subgraph Test_Data_Preparation ["Test Data Preparation"]
end

load["get_datasets()"]
testloader["DataLoader (no shuffle)"]
model["AnalyticalPointNetLK"]
extract["Feature Extraction"]
iter["Iterative Refinement"]
metrics["Compute Metrics"]
results["Results Logging"]

    load --> testloader
    testloader --> model
    model --> extract
    extract --> iter
    iter --> metrics
    metrics --> results
```

Sources: [test.py L87-L106](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/test.py#L87-L106)

 [test.py

106](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/test.py#L106-L106)

### 4.1 Detailed Testing Pipeline

In the testing phase, the system processes each point cloud pair through the following steps:

1. Load point clouds (either directly or through voxelization)
2. Extract features using the PointNet backbone
3. Iteratively estimate transformation using the Lucas-Kanade algorithm
4. Evaluate transformation accuracy using metrics such as rotation error and translation error

```mermaid
sequenceDiagram
  participant Data
  participant Model
  participant Features
  participant LK
  participant Eval

  Data->Model: Source & Target Point Clouds
  Model->Features: Extract Features
  Features->LK: Feature Jacobian Calculation
  LK->Eval: Iterative Pose Refinement
```

Sources: [test.py

106](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/test.py#L106-L106)

## 5. Complete Data Flow Pipeline

The following diagram illustrates the end-to-end data flow from input data to final transformation estimation:

```mermaid
flowchart TD

subgraph OutputEvaluation ["Output/Evaluation"]
end

subgraph Transformation_Estimation ["Transformation Estimation"]
end

subgraph Feature_Extraction ["Feature Extraction"]
end

subgraph Data_Loading__Preprocessing ["Data Loading & Preprocessing"]
end

subgraph Input_Data ["Input Data"]
end

raw["Raw Dataset"]
cat["Category Files"]
load["Dataset Loading"]
transform["Data Transformation"]
voxelize["Voxelization (for real data)"]
augment["Data Augmentation (for training)"]
ptnet["PointNet Feature Extraction"]
feats["Global Point Cloud Features"]
jacobian["Feature Jacobian Calculation"]
iclk["Iterative Lucas-Kanade"]
update["Pose Update"]
est["Estimated Transformation"]
metrics["Evaluation Metrics"]
viz["Visualization (optional)"]

    raw --> load
    cat --> load
    load --> transform
    transform --> voxelize
    transform --> augment
    voxelize --> ptnet
    augment --> ptnet
    ptnet --> feats
    feats --> jacobian
    jacobian --> iclk
    iclk --> update
    update --> iclk
    update --> est
    est --> metrics
    est --> viz
```

Sources: [data_utils.py](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py)

 [train.py](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/train.py)

 [test.py](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/test.py)

## 6. Data Flow During Voxel-based Registration

For 3DMatch dataset and real-world point clouds, voxel-based registration involves additional processing:

```mermaid
flowchart TD

subgraph Output ["Output"]
end

subgraph Transformation ["Transformation"]
end

subgraph Voxelization_Process ["Voxelization Process"]
end

subgraph Input_Point_Clouds ["Input Point Clouds"]
end

p0_pre["Raw Source Point Cloud"]
p1_pre["Raw Target Point Cloud"]
down["Voxel Downsampling"]
overlap["Find Overlapping Regions"]
voxelize["Convert Points to Voxels"]
coords["Calculate Voxel Coordinates"]
intersect["Find Intersecting Voxels"]
transform["Apply Transformation"]
voxels_p0["Voxelized Source Points"]
voxels_p1["Voxelized Target Points"]
coords_p0["Source Voxel Coordinates"]
coords_p1["Target Voxel Coordinates"]
igt["Ground Truth Transform"]

    down --> overlap
    overlap --> voxelize
    voxelize --> coords
    coords --> intersect
    intersect --> coords
    transform --> igt
```

Sources: [data_utils.py L55-L179](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py#L55-L179)

 [data_utils.py L36-L52](https://github.com/Lilac-Lee/PointNetLK_Revisited/blob/4c5fbb1a/data_utils.py#L36-L52)

This voxelization-based data flow is particularly important for handling real-world data with partial overlaps and sensor noise, ensuring that the Lucas-Kanade algorithm operates on well-structured and matched regions of the point clouds.