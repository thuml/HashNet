#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pairwise_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
void PairwiseLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter pairwise_param(this->layer_param_);
  pairwise_param.set_type("Pairwise");
  class_num_ = this->layer_param_.pairwise_param().class_num();
  l_threshold_ = this->layer_param_.pairwise_param().l_threshold();
  q_threshold_ = this->layer_param_.pairwise_param().q_threshold();
  l_lambda_ = this->layer_param_.pairwise_param().l_lambda();
  q_gamma_ = this->layer_param_.pairwise_param().q_gamma();
  sigmoid_param_ = this->layer_param_.pairwise_param().sigmoid_param();
  continous_similarity_ = this->layer_param_.pairwise_param().continous_similarity();
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  pairwise_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.pairwise_param().axis());
  outer_num_ = bottom[0]->count(0, pairwise_axis_);
  inner_num_ = bottom[0]->count(1);
  label_dim_ = bottom[1]->count(1);
  product_.Reshape(1, 1, bottom[0]->count(0, 1), bottom[0]->count(0, 1));
  loss_.Reshape(1, 1, bottom[0]->count(0, 1), bottom[0]->count(0, 1));
  count_.Reshape(1, 1, outer_num_, outer_num_);
  own_similarity_.Reshape(1, 1, outer_num_, outer_num_);
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PairwiseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(PairwiseLossLayer);
#endif

INSTANTIATE_CLASS(PairwiseLossLayer);
REGISTER_LAYER_CLASS(PairwiseLoss);

}  // namespace caffe
