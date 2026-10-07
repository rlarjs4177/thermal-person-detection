# Repository contents

This repository contains the source code and configuration files for the
thermal person-detection experiments.

Datasets, sample images, trained weights, checkpoints, generated COCO
metadata, virtual environments, caches, and experiment outputs are omitted.
Provide the dataset locally using the layout documented in `README.md`:

```text
data/
  train/{image,gt}/
  val/{image,gt}/
  test/{image,gt}/
```

Pretrained and trained model weights must also be obtained or generated
separately. Do not commit credentials or private datasets.

## Third-party code

`third_party/RT-DETR` is derived from
https://github.com/lyuwenyu/RT-DETR and retains its Apache-2.0 `LICENSE`.
No license is asserted here for the repository's original project code.
