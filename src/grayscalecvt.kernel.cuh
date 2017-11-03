
#ifndef GRAYSCALECVT_KERNEL_H_
#define GRAYSCALECVT_KERNEL_H_

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
__global__ void convertToGrayscale(cudaTextureObject_t inputImgTex, uchar* outputImg,
                                  int iWidth)
{
  //--- access thread id ---
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int tidy = blockDim.y * blockIdx.y + threadIdx.y;


  //--- pixel variables ---
  unsigned int texel;     // assign memory to download texel to
  uchar* colPx = (uchar*) &texel;  // assign memory pointer to pixel memory uchar[4]
  uchar graycalePx;

  //--- read data from texture ---
  texel = tex2D<unsigned int>(inputImgTex, tidx, tidy);

  graycalePx = (uchar)(0.2126f * (float)colPx[0]); // R
  graycalePx += (uchar)(0.7152f * (float)colPx[1]); // G
  graycalePx += (uchar)(0.0722f * (float)colPx[2]); // B

  //--- write to output ---
  outputImg[tidy * iWidth + tidx] = graycalePx;
}

#endif // #ifndef GRAYSCALECVT_KERNEL_H_
