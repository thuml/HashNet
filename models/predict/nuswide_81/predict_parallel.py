import numpy as np
import scipy as sp
import sys
import caffe
import os.path as opath
from multiprocessing import Pool 
import argparse

def save_code_and_label(params):
    database_code = params['database_code']
    validation_code = params['validation_code']
    database_labels = params['database_labels']
    validation_labels = params['validation_labels']
    path = params['path']
    np.save(opath.join(path, "database_code.npy"), database_code)
    np.save(opath.join(path, "database_label.npy"), database_labels)
    np.save(opath.join(path, "validation_code.npy"), validation_code)
    np.save(opath.join(path, "validation_label.npy"), validation_labels)

def mean_average_precision(params):
    database_code = params['database_code']
    validation_code = params['validation_code']
    database_labels = params['database_labels']
    validation_labels = params['validation_labels']
    R = params['R']
    query_num = validation_code.shape[0]

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []
    ground_truth = np.sum(np.dot(database_labels, validation_labels.T), axis=0)
    validation_labels[validation_labels==0] = -1
    for i in range(query_num):
        label = validation_labels[i, :]        
        if ground_truth[i] > 0:
            idx = ids[:, i]
            imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
            relevant_num = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, R+1, 1)
            if relevant_num != 0:
                APx.append(np.sum(Px * imatch) / relevant_num)
            else:
                APx.append(np.sum(Px * imatch) / 1)            
    
    return np.mean(np.array(APx))
        
def get_codes_and_labels(params):
    device_id = int(params["gpu"])
    caffe.set_device(device_id)
    caffe.set_mode_gpu()
    model_file = params['model_file']
    pretrained_model = params['pretrained_model']
    dims = params['image_dims']
    scale = params['scale']
    database = open(params['database'], 'r').readlines()
    validation = open(params['validation'], 'r').readlines()
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

    cur_pos = 0
    while 1:
        lines = validation[cur_pos : cur_pos + batch_size]
        if len(lines) == 0:
            break;
        cur_pos = cur_pos + len(lines)
        images = [caffe.io.load_image(line.strip().split(" ")[0]) for line in lines]
        labels = [[int(i) for i in line.strip().split(" ")[1:]] for line in lines]

        codes = net.predict(images, oversample=False)
        #codes = net.predict(images)
        [validation_code.append(c) for c in codes]
        [validation_labels.append(l) for l in labels]
        
        print str(cur_pos) + "/" + str(len(validation))
        if len(lines) < batch_size:
            break;
        
    return dict(database_code=np.sign(np.array(database_code)), database_labels=np.array(database_labels), validation_code=np.sign(np.array(validation_code)), validation_labels=np.array(validation_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Hash Code.')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='?', default="0",
                    help='running gpu id')
    parser.add_argument('--model_path', type=str, nargs="?",                    
                    help='the caffemodel path')
    parser.add_argument('--save_path', type=str, nargs='?',                    
                    help='the path to save predict code')
    parser.add_argument('--code_path', type=str, nargs='?', default="",
                    help='the predicted hash code path')

    args = parser.parse_args()


    if args.code_path:
        code_and_label = {}
        code_and_label["database_code"] = np.load(opath.join(args.code_path, "database_code.npy"))
        code_and_label['database_labels'] = np.load(opath.join(args.code_path,"database_label.npy"))
        code_and_label["validation_code"] = np.load(opath.join(args.code_path, "validation_code.npy"))
        code_and_label['validation_labels'] = np.load(opath.join(args.code_path, "validation_label.npy"))
    else:
        nthreads = 8
        params = []
        for i in range(nthreads):
            params.append(dict(model_file="./models/predict/deploy.prototxt",
                  pretrained_model=args.model_path,
                  image_dims=(256,256),
                  scale=255,
                  database="./data/nuswide_81/test/database" + str(i) + ".txt",
                  validation="./data/nuswide_81/test/test" + str(i) + ".txt",
                  batch_size=50,
                  mean_file="./python/caffe/imagenet/ilsvrc_2012_mean.npy",
                  gpu=args.gpu))

        pool = Pool(nthreads)
        results = pool.map(get_codes_and_labels, params)
        print("start combine")
        code_and_label = results[0]
        for i in range(1, nthreads):
            code_and_label['database_code'] = np.vstack([code_and_label["database_code"], results[i]['database_code']])
            code_and_label['database_labels'] = np.vstack([code_and_label["database_labels"],results[i]['database_labels']])
            code_and_label['validation_code'] = np.vstack([code_and_label["validation_code"], results[i]['validation_code']])
            code_and_label['validation_labels'] = np.vstack([code_and_label["validation_labels"], results[i]['validation_labels']])

        print("saving ...")
        code_and_label['path'] = str(args.save_path)
        save_code_and_label(code_and_label)
        print("saving done")

print("start predict")
code_and_label['R'] = 5000
mAP = mean_average_precision(code_and_label)
print ("MAP: "+ str(mAP))
