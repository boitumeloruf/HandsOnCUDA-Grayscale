# Hands-On CUDA - Grayscale Conversion

&copy; <i>2017 Boitumelo Ruf. All rights reserved.</i>

Example program performing a grayscale conversion on a GPU with CUDA. The CUDA kernel expects a ```cudaTextureObject_t``` with 4 bytes per pixel, which hold the color coded in RGBA. It uses an unsigned int to fetch the pixel data from the texture and references each channel as a uchar array of size 4. After computing the grayscale value it saves the result into the data array of the output image.

On host side the program uses the ```cv::Mat``` container of OpenCV to load, store and visualize the input and output images. The program incoorporates a custom function that uploads the ```cv::Mat``` into a ```cudaTextureObject_t``` and downloads the ```uchar``` output array into a ```cv::Mat```.

Note that it is common practise in the context of GPGPU to use a vectorized image data where each pixel holds 4 bytes of image data, i.e. RGBA. Since OpenCV by default uses the BGR format it is required to pretransform the image from BGR to RGBA.

## Dependencies

This program is developed and tested on Ubuntu 16.04 with GCC 5.4 and depends on following 3rd party libraries:

- Qt >5.0
- OpenCV >3.0
- CUDA >7.5

## Getting Started

- Download or clone repository.
- Copy ```src/cuda.pri.tpl``` and ```src/opencv.pri.tpl``` into ```src/cuda.pri``` and ```src/opencv.pri```.
- Replace tags ```<edit-here>``` in ```src/cuda.pri``` and ```src/opencv.pri``` with corresponding information such as version number and root directory of OpenCV and CUDA installations.
- Open ```CUDA-GrayscaleCvt.pro``` with QtCreator or run ```qmake CUDA-GrayscaleCvt.pro```.
- Build and run with image as command line argument. ```CUDA-GrayscaleCvt ../assets/tsukuba.png```. Build output is located in bin/.

## Expected Results

<b>Input</b> [Martull et al. (2012)]: <br>
![alt text][input]

<b>Output</b>: <br>
![alt text][output]

[input]: assets/tsukuba.png
[output]: assets/tsukuba_grayscale.png

## References

<b> Martull et al. (2012)</b><br>
Martull, S.; Peris, M. & Fukui, K. "Realistic cg stereo image dataset with ground truth disparity maps", <i>Proceedings of the International Conference on Pattern Recognition (ICPR)</i>, 2012, vol. 111, pp. 117-118.
