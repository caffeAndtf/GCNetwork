import numpy as np
from src.parse_arguments import *
from src.gcnetwork import *
import glob
import os
import psutil
from src.generator import *


def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()


def _predictFromArrays_(model, left, right, bs):
    return model.predict([left, right], bs)


def _predictFromGenerator_(model, generator, steps, max_q_size):
    return model.predict_generator(generator, steps, max_q_size)


def Predict():
    hp, tp, up, env = parseArguments()
    pspath = tp['pspath']
    ext = up['file_extension']
    data_path = "./testcase"
    bs = tp['batch_size']
    max_q_size = tp['max_q_size']
    verbose = tp['verbose']

    def get_session(gpu_fraction=0.95):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    #K.set_session(get_session())
    model = createGCNetwork(hp, tp, True)
    if data_path.endswith('npz'):
        images = np.load(data_path)
        print "Predict data using arrays"
        pred = _predictFromArrays_(model, images[1], images[2], bs)
        np.save(pspath, pred)
    else:
        left_path = os.path.join(data_path, 'left')
        right_path = os.path.join(data_path, 'right')
        left_images = glob.glob(left_path + "/*.{}".format(ext))
        right_images = glob.glob(right_path + "/*.{}".format(ext))
        generator = generate_arrays_from_file(left_images, right_images, up)
        print "Predict data using generator..."
        pred = model.predict_generator(generator, max_queue_size=max_q_size, steps=bs, verbose=verbose)
        print pred[0]
        np.save(os.path.join(data_path, "rsl.npy"), pred)
    K.clear_session()


if __name__ == "__main__":
    Predict()
