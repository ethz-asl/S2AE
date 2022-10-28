# Spherical Semantic Segmentation

Semantic segmentation for robotic systems can enable a wide range of applications, from self-driving cars to augmented reality systems.
LiDAR scans can have various different characteristics and properties, such as number of beams, vertical FoV, angular resolution. 
Our method provides a framework to seamlessly train on pointcloud data from various different LiDAR sensor models and types.

## Installation

S2AE was written using __PyTorch__ ([http://pytorch.org/](http://pytorch.org/)) and depends on a few libraries.
  * __s2cnn__: [https://github.com/jonas-koehler/s2cnn](https://github.com/jonas-koehler/s2cnn)
  * __lie_learn__: [https://github.com/AMLab-Amsterdam/lie_learn](https://github.com/AMLab-Amsterdam/lie_learn)
  * __pynvrtc__: [https://github.com/NVIDIA/pynvrtc](https://github.com/NVIDIA/pynvrtc)

Submodule references to these repositories can be found in the `deps` folder

## Reference

Our paper is available at

*Bernreiter, Lukas, Lionel Ott, Roland Siegwart, and Cesar Cadena. "SphNet: A Spherical Network for Semantic Pointcloud Segmentation"* [[ArXiv](https://arxiv.org/abs/2210.13992)]
