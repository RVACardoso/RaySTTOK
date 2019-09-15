# RaySTTOK

RaySTTOK is essentially a Python class which allows easy ray-tracing simulations within the environment of ISTTOK's tomography system. It implements several possible radiation sources and allows the simulation of their corresponding detector measurements.
RaySTTOK also allows the computation of ISTTOK's projection matrix, taking into account the detector's complex viewing geometries and reflections. This matrix can mimic the results of the ray-tracing simulation at a fraction of the computational cost.

![view cone image](figures/capa.png?raw=true "Title")

### Prerequisites

RaySTTOK uses the heavy machinery implemented on the Raysetc Python package. Thus, the main prerequisite is:

```
Raysect
```
Other required packages include:
```
Numpy
Matplotlib
Scipy
Random
```

### Installing

The first step to successfuly use RaySTTOK is the installation of the package [Raysect](https://raysect.github.io/documentation/installation.html).
Then, simply place the file "raysttok.py" and the folder "resources" on the desired working directory.

## Getting Started

The simulation of a synthetic Gaussian emissivity profile can be done with:
```
raysttok = RaySTTOK(reflections=True, pixel_samples=100)

raysttok.place_plasma(shape='gaussian', emissiv_gain=1e3, mean=[0.05, -0.05], cov=[[5e-4, 0], [0, 5e-4]])

raysttok.simulate_rays()
raysttok.plot_detectors()
raysttok.show_plots()
```

On the other hand, the computation of ISTTOK's projection matrix can be performed with:
```
raysttok = RaySTTOK(reflections=True, pixel_samples=10)
raysttok.get_proj_matrix(pixel_side=15, out_file="proj_matrix1")
raysttok.plot_line_matrix(line=0, mat_file="proj_matrix1.npy")
raysttok.show_plots()
```
A careful description of the implemented methods and their potential can be found on [RaySSTOK Wiki](https://github.com/RVACardoso/RaySTTOK/wiki/RaySTTOK-Wiki)

## Authors

* **R. V. A. Cardoso**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
