# Training Instructions
## Different hash code length
You can set different code length in the train_val.prototxt file as follows.
First, find the encoding layer:
```
layer {
  name: "fc8_flickr"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8_code"
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 48
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
```
Set the num_output to the code length you want.
And then find the pairwise_loss layer:
```
layer {
  name: "pairwise_loss"
  type: "PairwiseLoss"
  bottom: "fc8_code_part1"
  bottom: "label1"
  bottom: "fc8_code_part2"
  bottom: "label2"
  top: "pairwise_loss"
  pairwise_param {
    class_num: 100.0
    l_threshold: 15
    q_threshold: 15
    l_lambda: 1
    sigmoid_param: 0.2 # 10 / the length of code
    continous_similarity: false
  }
}
```
Set the sigmoid_param to 10 / code length.

## Different datasets
All the parameters are set for the datasets in our experiments. If you want to perform our methods on your own dataset, you need to calculate (the sum of number of 0s and number of 1s) / the number of 1s in your train set and set class_num parameter in pairwise_loss layer.
