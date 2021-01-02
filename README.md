# YOLOv3 - Julia Version w/ Knet
This repository hosts Julia code for YOLOv3.

```
@article{redmon2018yolov3,
  title={Yolov3: An incremental improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal={arXiv preprint arXiv:1804.02767},
  year={2018}
}
```

By using code in this repo you can:
- Train a Darknet network
- Use YOLOv3 (Darknet) architecture to detect and classify objects.

Before using any of the code below, please load project environment by runing following code snippet:
```
import Pkg
Pkg.activate("Project.toml")
```

## File Descriptions
### models.jl
In this file you will find code for creating/loading/saving Darknet architecture. Also, methods for loss calculation and object detection inference.

```
include("models.jl")
```

__Model Creation__
```
darknet = Darknet(<model_cfg_file>; img_size=(416, 416))
```

__Save/Load a Model__
```
# To save:
save_model(darknet, <model_jld2_filepath>)

# To load:
darknet = load_model(<model_jld2_filepath>)
```

__Inference__
```
bbox_predictions = darknet(<image_float32_array>; training=false)
```

__Loss__
```
loss = darknet(<image_float32_array>, <label_array>)
```

__(Mean) Average Precision__
```
include("utils/map.jl")
mAP, _ = compute_mAP(bbox_predictions, <label_array>)
```

### train.jl
Training tool for YOLOv3. You can see the available training parameter options by running:
```
julia train.jl --help

###
Output:
arguments:
  --model-out MODEL-OUT
                        Model file path to save the trained model.
  --results RESULTS     Result file. (default: "results.jld2")
  --model-config MODEL-CONFIG
                        Network config file. (default:
                        "cfg/yolov3.cfg")
  --preload PRELOAD     Pre-trained model file.
  --trndata TRNDATA [TRNDATA...]
                        COCO2014Data files for training.
  --valdata VALDATA [VALDATA...]
                        COCO2014Data files for validation.
  --epoch EPOCH         Number of epochs. (type: Int64, default: 100)
  --iepoch IEPOCH       Number of instance epochs. (type: Int64,
                        default: 2)
  --lr LR               Learning rate (type: Float64, default: 0.001)
  --period PERIOD       Status printing period. (type: Int64, default:
                        10)
  --bs BS               Batch size (type: Int64, default: 8)

```

### coco2014.jl
Cotains all necessary functions to prepare and process coco2014 object detection data. For detailed usage instructions, please see `examples/data_operations.ipynb`.

### nn.jl
Contains all Knet layer definitions that can be used to build a network from scratch such as Darknet. Layers include YOLOLayers, FeatureFusions, FeatureConcat, etc.

```
include("nn.jl")
import .NN

# Start to call NN.<layer> for usage()
```

## Visual Tools
```
Please refer to utils/img.jl.
```
