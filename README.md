# The algorithms compared with [PaT](https://github.com/JeepWay/PaT)

## Installation
Please follow the [instructions](https://github.com/JeepWay/PaT?tab=readme-ov-file#2-installation) in the [PaT](https://github.com/JeepWay/PaT) to install the necessary dependencies.

## Algorithms
1. Shelf Next Fit
2. Shelf First Fit
3. Skyline Bottom-Left
4. Zhao-2D
5. Deep-Pack
6. Zhao-2D with Truth Mask
7. Deep-Pack with Truth Mask

## Usage example
If you want run deep-pack, please moving to the deep-pack directory.
```bash
cd deep-pack
```
Run the following command to start the deep-pack training and testing:
```bash
python main.py --config_path settings/v1_deeppack_DDQN-h200-rC.yaml
```

Like Pat, the complete commands for all experiments in the thesis are included in the `scripts/all.sh` script under each algorithm directory.

For example, the path to  `scripts/all.sh` for deep-pack is [deep-pack/scripts/all.sh](deep-pack/scripts/all.sh).

## References
* [A Thousand Ways to Pack the Bin – A Practical Approach to Two-Dimensional Rectangle Bin Packing](https://raw.githubusercontent.com/rougier/freetype-gl/master/doc/RectangleBinPack.pdf)
* [GitHub: RectangleBinPack](https://github.com/juj/RectangleBinPack)
* [Online 3D Bin Packing with Constrained Deep Reinforcement Learning](https://arxiv.org/abs/2006.14978)
* [GitHub: Online-3D-BPP-DRL](https://github.com/alexfrom0815/Online-3D-BPP-DRL)
* [Deep-Pack: A Vision-Based 2D Online Bin Packing Algorithm with Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/8956393)
* [GitHub: Deep-Pack (unofficial implementation)](https://github.com/JeepWay/DeepPack)
