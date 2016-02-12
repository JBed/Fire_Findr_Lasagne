import cPickle
import os
import errno
import itertools
import termios
import sys

from sklearn import metrics

from bunch import Bunch
import matplotlib
from sklearn.metrics import auc

import ml_utils

if 'LOCATION' in os.environ and os.environ['LOCATION'] == 'aws':
    matplotlib.use('svg')

import theano
import numpy as np
import theano.tensor as T
from matplotlib import pyplot as plt

def describe_device(dev):
    print dev.name(), dev.id, dev.pci_bus_id(), dev.compute_capability(), dev.total_memory()


def memory_usage():
    # Taken from http://stackoverflow.com/questions/897941/python-equivalent-of-phps-memory-get-usage
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])
    finally:
        if status is not None:
            status.close()
    return result




def set_numpy_print_opts():
    np.set_printoptions(threshold=300000)
    np.set_printoptions(precision=10)
    np.set_printoptions(suppress=True)


def draw_roc_curve(y_true, y_pred, filepath='out.svg'):
    print y_true, y_pred, y_true * y_pred
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='lower right')
    plt.savefig(filepath)


def detect_nan(i, node, fn):
    for output in fn.outputs:
        if np.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break



def serialize(obj, path):
    f = file(path, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def deserialize(path):
    return cPickle.load(file(path, 'rb'))


def dump_model(model, model_path):
    # Dumping the model to file
    print ('Dumping model to ' + model_path)
    f = file(model_path, 'wb')

    # We dump only the parameters
    cPickle.dump(model.__getstate__(), f, protocol=cPickle.HIGHEST_PROTOCOL)



def set_theano_fast_compile():
    theano.config.mode = 'FAST_COMPILE'


def set_theano_fast_run():
    theano.config.mode = 'FAST_RUN'


def set_theano_debug():
    theano.config.mode = 'DebugMode'


def theano_compilation_mode():
    return theano.config.mode


def mark(tensor):
    tensor.name = 'marked'

