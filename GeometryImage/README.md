# SurfacePlaneMapper
SurfacePlaneMapper is a module to flatten a surface (such as brain surface or hippocampus) into a 2D image. This functionality allows us to utilize various deep learning architectures, such as ResNet and EfficientNet, which are commonly used for 2D image classification, with surface data. This is implemented using a technique called `Geometry image` proposed by Gu et al. (https://hhoppe.com/gim.pdf) where geometry is resampled into a completely regular 2D grid. The process involves cutting the surface into a disk using a network of cut paths, and then mapping the boundary of this disk to a square. Both geometry and other signals are stored as 2D grids, with grid samples in
implicit correspondence.
![](geom_image_intro.png?raw=true)

## Input
![](geom_image.png?raw=true)
* Input Directory: The module assumes that the `input directory` is organized in one of the following two manners:
1.
```input_directory
| ---- subject_id
       | ---- session
              | ---- modality
                     | ---- *.txt
```
The `txt` files contain the features at each point on the surface.
2.
```
input_directory
| ---- subject_id
       | ---- session
              | ---- *.vtk
```
* Sphere Template: A `vtk` / `obj` file containing the icosahedron subdivision of the shpere
  where the surface is mapped.
* Output Directory: The output directory where the results will be saved. The module will create
  similar folder hierarchy as the input directory. 
```
output directory
| ---- subject_id
       | ---- session
              | ---- modality
                     | ---- *.png
```
  The results will be saved as a png file with 512x512 resolution.
