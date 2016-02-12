import sys
import logging

import numpy as np

from sklearn.cross_validation import ShuffleSplit
# from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold


class TrainSplit(object):
    def __init__(self, eval_size, stratify=True, random_state=None):
        self.eval_size = eval_size
        self.stratify = stratify
        self.random_state = random_state

    def __call__(self, X, y, net):
        if self.eval_size is not None:
            if net.regression or not self.stratify:
                # test_size = self.eval_size
                # kf = ShuffleSplit(
                #     y.shape[0], test_size=test_size,
                #     random_state=self.random_state
                # )
                # train_indices, valid_indices = next(iter(kf))
                # valid_indices = shuffle(valid_indices)
                test_size = 1 - self.eval_size
                kf = ShuffleSplit(
                    y.shape[0], test_size=test_size,
                    random_state=self.random_state
                )
                valid_indices, train_indices = next(iter(kf))
            else:
                n_folds = int(round(1 / self.eval_size))
                kf = StratifiedKFold(y, n_folds=n_folds, random_state=self.random_state)
                train_indices, valid_indices = next(iter(kf))

            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = X[len(X):], y[len(y):]

        return X_train, X_valid, y_train, y_valid


def add_padding_to_bbox(x, y, w, h, pad, max_x, max_y, format='ltwh'):
    l, t = x, y
    r, b = l + w, t + h
    pad_x = int(round(w * pad / 2))
    pad_y = int(round(h * pad / 2))
    new_l = min(max(l - pad_x, 0), max_x)
    new_r = min(max(r + pad_x, 0), max_x)
    new_t = min(max(t - pad_y, 0), max_y)
    new_b = min(max(b + pad_y, 0), max_y)
    new_w = new_r - new_l
    new_h = new_b - new_t

    if format == 'ltwh':
        return int(new_l), int(new_t), int(new_w), int(new_h)
    elif format == 'ltrb':
        return int(new_l), int(new_t), int(new_r), int(new_b)
    else:
        raise ValueError('Format is not recongized: %s' % format)


class LogFile(object):
    """
    File-like object to log text using the `logging` module.
    http://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python/2216517#2216517
    """
    def __init__(self, name=None, stdout=None):
        self.logger = logging.getLogger(name)
        self.stdout = stdout

    def write(self, msg, level=logging.DEBUG):
        self.logger.log(level, msg)

        if self.stdout is not None:
            self.stdout.write(msg)

    def flush(self):
        # for handler in self.logger.handlers:
        #     handler.flush()

        if self.stdout is not None:
            self.stdout.flush()


def mirror_to_log(fname):
    logging.basicConfig(level=logging.DEBUG, filename=fname)

    # Redirect stdout
    sys.stdout = LogFile(fname, sys.stdout)


def float32(k):
    return np.cast['float32'](k)


def transformation(img, spec, perturb):
    # inter_size = spec['inter_size']
    mean = spec['mean']
    std = spec['std']
    img = img.astype(dtype=floatX)
    # img /= 255.0

    def apply_mean_std(img):
        if mean is not None:
            assert (len(mean) == spec['target_channels'])
            for channel in xrange(spec['target_channels']):
                img[:, :, channel] -= mean[channel]

        if std is not None:
            assert (len(std) == spec['target_channels'])
            for channel in xrange(spec['target_channels']):
                img[:, :, channel] /= std[channel]
        return img

    if perturb:
        img = perturb_fun(img, spec['augmentation_params'], target_shape=(spec['target_h'], spec['target_w']))

    # imgs.append(img)
    # img = np.copy(img)

    # PCA
    if spec['pca_data'] is not None:
        evs, U = spec['pca_data']
        ls = evs.astype(float) * np.random.normal(scale=spec['pca_scale'], size=evs.shape[0])
        noise = U.dot(ls).reshape((1, 1, evs.shape[0]))
        # print evs, ls, U
        # print 'noise', noise
        img += noise

    img = apply_mean_std(img)

    def f(img):
        img = np.rollaxis(img, 2)
        return img

    # The img was H x W x C before
    return f(img)

def print_traceback(f):
    def g(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print traceback.format_exc()
            raise
    return g


def find_bucket(s, buckets, wsp):
            if wsp < 0 or wsp >= s:
                return -1
            res = int(floor((wsp * buckets) / s))
            assert(res >= 0 and res < buckets)
            return res


def rev_find_bucket(s, buckets, wsp):
    return (wsp * s) / buckets

def rev_find_bucket_robert(s, buckets, wsp):
    raise RuntimeError()

