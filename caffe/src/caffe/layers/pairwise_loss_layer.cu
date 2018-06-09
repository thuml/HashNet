#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pairwise_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
__global__ void SimilarityProcess(const int nthreads, Dtype* similarity, Dtype label_dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if((similarity[index] < 0) || (similarity[index] >= label_dim)){
      //unknown label
      similarity[index] = Dtype(-1.0);
    }
    else if(similarity[index] > 0){
      //similar label
      similarity[index] = Dtype(1.0);
    }
  }
}

template <typename Dtype>
__global__ void ContinousSimilarityProcess(const int nthreads, const Dtype* similarity, const Dtype* similarity1, Dtype* similarity2, Dtype* sim, const int outer_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int data_id1 = index / outer_num;
    int data_id2 = index % outer_num;
    sim[index] = similarity[index] * similarity[index] / (similarity1[outer_num*data_id1+data_id1] * similarity2[outer_num*data_id2+data_id2]);
    if(sim[index] == 0){
      sim[index] = 0.25;
    }
  }
}

template <typename Dtype>
__global__ void RemoveZero(const int nthreads, Dtype* similarity1, Dtype* similarity2) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(similarity1[index] == 0){
      similarity1[index] = 1.0;
    }
    if(similarity2[index] == 0){
      similarity2[index] = 1.0;
    }
  }
}


template <typename Dtype>
__global__ void PairwiseLossForwardGPU(const int nthreads, const int num, const Dtype* similarity, 
       const Dtype* exp_product, const Dtype* product, const Dtype threshold, Dtype* count, Dtype* loss_data, const Dtype class_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(similarity[index] >= 0){
      count[index] = Dtype(1.0);
      if((threshold >= 0) && (product[index] >= threshold)){
        loss_data[index] = product[index] * (1 - (similarity[index] > 0));
      }
      else{
        loss_data[index] = log(1 + exp_product[index]) - (similarity[index] > 0) * product[index];
      }
      if(similarity[index] > 0){
        loss_data[index] = loss_data[index] * class_num;
        count[index] *= class_num;
      }
    }
    else{
      count[index] = Dtype(0.0);
      loss_data[index] = Dtype(0.0);
    }
  }
}


template <typename Dtype>
void PairwiseLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* similarity = loss_.mutable_gpu_data();
  Dtype* dot_product = product_.mutable_gpu_data();
  Dtype* exp_product = product_.mutable_gpu_diff();
  Dtype* loss_data = loss_.mutable_gpu_diff();
  Dtype* count = count_.mutable_gpu_data();
  Dtype* similarity1 = own_similarity_.mutable_gpu_data();
  Dtype* similarity2 = own_similarity_.mutable_gpu_diff();

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_data1 = bottom[2]->gpu_data();
  Dtype* label = bottom[1]->mutable_gpu_data();
  Dtype* label1 = bottom[3]->mutable_gpu_data();

  int nthreads = outer_num_ * outer_num_;

  Dtype loss, count_num;
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, label_dim_, 
      Dtype(1.0), label, label1, Dtype(0.0), similarity);
  if (continous_similarity_){
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, label_dim_, 
      Dtype(1.0), label, label, Dtype(0.0), similarity1);
    caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, label_dim_, 
      Dtype(1.0), label1, label1, Dtype(0.0), similarity2);

    RemoveZero<Dtype><<<CAFFE_GET_BLOCKS(own_similarity_.count()), 
      CAFFE_CUDA_NUM_THREADS>>>(own_similarity_.count(), similarity1, similarity2);

    ContinousSimilarityProcess<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, similarity, similarity1, similarity2, loss_data, outer_num_);
    caffe_gpu_memcpy(nthreads*sizeof(Dtype), loss_data, similarity1);
  }

  SimilarityProcess<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, similarity, label_dim_);
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, outer_num_, outer_num_, inner_num_, 
      Dtype(1.0), bottom_data, bottom_data1, Dtype(0.0), dot_product);
  caffe_gpu_scal(outer_num_ * outer_num_, sigmoid_param_, dot_product);
  caffe_gpu_exp(outer_num_ * outer_num_, dot_product, exp_product);
  
  PairwiseLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, similarity, exp_product, 
      dot_product, l_threshold_, count, loss_data, class_num_); 
  caffe_gpu_asum(nthreads, loss_data, &loss);
  caffe_gpu_asum(nthreads, count, &count_num);
  loss /= (count_num > 0 ? count_num : Dtype(1));
  LOG(INFO) << "L loss:" << loss;
  loss = loss * (l_lambda_ > 0);
  top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
__global__ void PairwiseLossBackwardGPU(const int nthreads, const int num, 
          const Dtype* similarity, const Dtype* exp_product, Dtype* count, Dtype* diff, const Dtype class_num) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      if(similarity[index] >= 0){
          diff[index] = 2 * (
              1 / (1 + 1 / exp_product[index]) - 
              (similarity[index] > 0)
          );
          count[index] = Dtype(1.0);
          if(similarity[index] > 0){
              diff[index] = diff[index] * class_num;
              count[index] *= class_num;
          }
      }
      else{
          diff[index] = Dtype(0.0);
          count[index] = Dtype(0.0);
      }
  }
}


template <typename Dtype>
void PairwiseLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* diff = count_.mutable_gpu_diff();
    Dtype* count = count_.mutable_gpu_data();
    const Dtype* similarity = loss_.gpu_data();
    const Dtype* exp_product = product_.gpu_diff();

    const Dtype* similarity1 = own_similarity_.gpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff1 = bottom[2]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_data1 = bottom[2]->gpu_data();
    int nthreads = outer_num_ * outer_num_;
    
    //calculate diff
    PairwiseLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, similarity,
        exp_product, count, diff, class_num_);
        
    if(continous_similarity_){
      caffe_gpu_mul(nthreads, diff, similarity1, diff);
      caffe_gpu_scal(nthreads, Dtype(4), diff);
    }
    //copy to bottom_diff
    Dtype count_num;
    caffe_gpu_asum(nthreads, count, &count_num);
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, inner_num_, outer_num_, 
        l_lambda_ / (count_num > 0 ? count_num : Dtype(1)), diff, bottom_data1, 
        Dtype(0.0), bottom_diff); 
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, outer_num_, inner_num_, outer_num_, 
        l_lambda_ / (count_num > 0 ? count_num : Dtype(1)), diff, bottom_data, 
        Dtype(0.0), bottom_diff1);
    caffe_gpu_scal(outer_num_, sigmoid_param_, bottom_diff);
    caffe_gpu_scal(outer_num_, sigmoid_param_, bottom_diff1);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PairwiseLossLayer);

}  // namespace caffe
