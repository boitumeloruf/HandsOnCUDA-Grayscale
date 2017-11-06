////////////////////////////////////////////////////////////////////////////////
//! Copyright 2017 Boitumelo Ruf. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_GRAYSCALECVT_CUH
#define CUDA_GRAYSCALECVT_CUH

// OpenCV
#include <opencv2/core.hpp>

/**
 * @brief Entry point to run grayscale conversion on CUDA.
 * @param[in] inputImg
 * @return output image
 */
cv::Mat runCudaGrayscaleCvt(const cv::Mat& inputImg);

#endif // CUDA_GRAYSCALECVT_CUH
