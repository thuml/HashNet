// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layers/scale_tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ScaleTanHForward(const int n, const Dtype* in, const Dtype scale, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = tanh(scale * in[index]);
  }
}

template <typename Dtype>
void ScaleTanHLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ScaleTanHForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, scale_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ScaleTanHBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype scale, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype tanhx = out_data[index];
    out_diff[index] = scale * in_diff[index] * (1 - tanhx * tanhx);
  }
}

template <typename Dtype>
void ScaleTanHLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ScaleTanHBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, scale_, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleTanHLayer);


}  // namespace caffe
