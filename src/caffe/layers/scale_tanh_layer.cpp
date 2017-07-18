// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layers/scale_tanh_layer.hpp"

namespace caffe {

template <typename Dtype>
void ScaleTanHLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  scale_ = this->layer_param_.scale_tanh_param().init_scale();
  gamma_ = this->layer_param_.scale_tanh_param().gamma();
  iter_ = 0;
  step_size_ = this->layer_param_.scale_tanh_param().step_size();
  power_ = this->layer_param_.scale_tanh_param().power();
  init_scale_ = this->layer_param_.scale_tanh_param().init_scale();
}

template <typename Dtype>
void ScaleTanHLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  iter_++;
  if(iter_ % step_size_ == 0) {
    scale_ = init_scale_ * pow((1+gamma_ * iter_), power_);
  }
}

template <typename Dtype>
void ScaleTanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanh(scale_ * bottom_data[i]);
  }
}

template <typename Dtype>
void ScaleTanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype tanhx;
    for (int i = 0; i < count; ++i) {
      tanhx = top_data[i];
      bottom_diff[i] = scale_ * top_diff[i] * (1 - tanhx * tanhx);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScaleTanHLayer);
#endif

INSTANTIATE_CLASS(ScaleTanHLayer);
REGISTER_LAYER_CLASS(ScaleTanH);

}  // namespace caffe
