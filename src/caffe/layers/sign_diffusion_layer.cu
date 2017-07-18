#include <algorithm>
#include <vector>

#include "caffe/layers/sign_diffusion_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GaussianForward(const int n, const Dtype* in, const Dtype scale, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = erf(in[index] * scale);
  }
}

template <typename Dtype>
void SignDiffusionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  GaussianForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, scale_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void GaussianBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype scale, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype tanhx = out_data[index] * scale;
    out_diff[index] = 1.1283791671 * in_diff[index] * exp(- (tanhx * tanhx)) * scale;
  }
}

template <typename Dtype>
void SignDiffusionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    GaussianBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom_data, scale_, bottom_diff);
   CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SignDiffusionLayer);


}  // namespace caffe
