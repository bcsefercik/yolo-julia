# YOLOv3 - Julia Version w/ Knet
This repository hosts Julia code for YOLOv3. (Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018)). By using code in this repo you can:
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
bboxes = darknet(<image_float32_array>; training=false)
```

__Loss__
```
loss = darknet(<image_float32_array>, <label_array>)
```

__Average Precision__
```
loss = darknet(<image_float32_array>, <label_array>; training=false)
```

### train.jl
Training tool for YOLOv3. You can see the available training parameter options by running:
```
julia train.jl --help
```

### coco2014.jl
Cotains all necessary functions to prepare and process coco2014 object detection data. For detailed usage instructions, please see `examples/data_operations.ipynb`.

### nn.jl
Contains all Knet layer definitions that can be used to build a network from scratch such as Darknet. Layers include YOLOLayers, FeatureFusions, FeatureConcat, etc.


## Visual Tools
```
TODO
```
