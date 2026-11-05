# Third-party notices

VRPR's own source code is under the [MIT License](LICENSE). Its dependencies have their own terms.

| Dependency | License | Used for |
| --- | --- | --- |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0, or a commercial license from Ultralytics | Plate detection and training |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Apache-2.0 | Text recognition |
| [OpenCV](https://github.com/opencv/opencv-python) | Apache-2.0 for OpenCV, with other terms for bundled components | Video and images |
| [NumPy](https://numpy.org) | BSD-3-Clause | Image arrays |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | Configuration |
| [Roboflow](https://github.com/roboflow/roboflow-python) | Apache-2.0 | Optional dataset download |

## AGPL-3.0 and distribution

The `ml` extra installs Ultralytics. The MIT License covers this repository's code, but it does not
remove AGPL-3.0 obligations from a product built on Ultralytics. If you distribute that product, or
offer it to others over a network, AGPL-3.0 can require you to publish the corresponding source
unless you hold a commercial license. Weights trained with Ultralytics may also be affected. Get
advice before using VRPR in a closed-source product.

## Removed code

Before version 1.0, this repository contained `Utils/sort.py`, a copy of
[SORT](https://github.com/abewley/sort) by Alex Bewley, under GPL-3.0. It is no longer part of the
source tree, but it remains in the git history.
