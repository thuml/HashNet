// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>


#include "caffe/layers/sign_diffusion_layer.hpp"
namespace caffe {

template <typename Dtype>
void SignDiffusionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  scale_ = this->layer_param_.sign_diffusion_param().init_scale();
  gamma_ = this->layer_param_.sign_diffusion_param().gamma();
  iter_ = 0;
  step_size_ = this->layer_param_.sign_diffusion_param().step_size();
  power_ = this->layer_param_.sign_diffusion_param().power();
  init_scale_ = this->layer_param_.sign_diffusion_param().init_scale();
}

template <typename Dtype>
void SignDiffusionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  iter_++;
  if(iter_ % step_size_ == 0) {
    scale_ = init_scale_ * pow((1+gamma_ * iter_), power_);
  }
}

template <typename Dtype>
void SignDiffusionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = erf(scale_ * bottom_data[i]);
  }
}

template <typename Dtype>
void SignDiffusionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype tanhx;
    for (int i = 0; i < count; ++i) {
      tanhx = bottom_data[i] * scale_;
      bottom_diff[i] = 1.1283791671 * top_diff[i] * exp(- tanhx * tanhx) * scale_;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SignDiffusionLayer);
#endif

INSTANTIATE_CLASS(SignDiffusionLayer);
REGISTER_LAYER_CLASS(SignDiffusion);

}  // namespace caffe
