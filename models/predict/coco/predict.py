import numpy as np
import scipy as sp
import sys
import caffe
from multiprocessing import Pool 

def save_code_and_label(params):
    database_code = np.array(params['database_code'])
    database_labels = np.array(params['database_labels'])
    path = params['path']
    np.save(path + "database_code.npy", database_code)
    np.save(path + "database_label.npy", database_labels)


        
def get_codes_and_labels(params):
    device_id = int(sys.argv[1])
    caffe.set_device(device_id)
    caffe.set_mode_gpu()
    model_file = params['model_file']
    pretrained_model = params['pretrained_model']
    dims = params['image_dims']
    scale = params['scale']
    database = open(params['database'], 'r').readlines()
    batch_size = params['batch_size']

    if 'mean_file' in params:
        mean_file = params['mean_file']
        net = caffe.Classifier(model_file, pretrained_model, channel_swap=(2,1,0), image_dims=dims, mean=np.load(mean_file).mean(1).mean(1), raw_scale=scale)
    else:
        net = caffe.Classifier(model_file, pretrained_model, channel_swap=(2,1,0), image_dims=dims, raw_scale=scale)
    
    database_code = []
    validation_code = []
    database_labels = []
    validation_labels = []
    cur_pos = 0
    
    while 1:
        lines = database[cur_pos : cur_pos + batch_size]
        if len(lines) == 0:
            break;
        cur_pos = cur_pos + len(lines)
        images = [caffe.io.load_image(line.strip().split(" ")[0]) for line in lines]
        labels = [[int(i) for i in line.strip().split(" ")[1:]] for line in lines]
        codes = net.predict(images, oversample=False)
        #codes = net.predict(images)
        [database_code.append(c) for c in codes]
        [database_labels.append(l) for l in labels]
        
        print str(cur_pos) + "/" + str(len(database))
        if len(lines) < batch_size:
            break;
        
    return dict(database_code=database_code, database_labels=database_labels)

nthreads = 1
params = []
base_dir = "/home/caozhangjie/cvpr_caffe/data/coco/"
file_list = []
for i in xrange(nthreads):
    file_list.append(base_dir+"train" + ".txt")

for i in range(nthreads):
    params.append(dict(model_file="/home/caozhangjie/cvpr_caffe/models/bvlc_reference_caffenet/deploy.prototxt",
                  pretrained_model="/home/caozhangjie/cvpr_caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel",
                  image_dims=(256,256),
                  scale=255,
                  database=file_list[i],
                  batch_size=50,
                  mean_file="/home/caozhangjie/cvpr_caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy"))

pool = Pool(nthreads)
results = pool.map(get_codes_and_labels, params)
print("start combine")
code_and_label = results[0]
for i in range(1, nthreads):
    [code_and_label['database_code'].append(c) for c in results[i]['database_code']]
    [code_and_label['database_labels'].append(c) for c in results[i]['database_labels']]

print("start save")
code_and_label['path'] = "./data/" + sys.argv[2]
save_code_and_label(code_and_label)
