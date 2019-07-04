# File utils
# ydawei@umich.edu

import os
import pickle
import datetime
import time
import argparse
import hashlib
import pickle

def save_obj(obj, name, verbal=False ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        if verbal:
            print " Done saving %s" % name


def load_obj(name, verbal=False):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
        if verbal:
            print " Done loading %s" % name
        return obj


def exec_print(cmd):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] %s' % (st, cmd))
    os.system(cmd)


def list_all_files_w_ext(folder, ext, recursive=False, cache=False, full_path=True):
    folder = os.path.abspath(folder)
    FILENAME_CACHE_DIR = 'cache_filename'
    hash_obj = hashlib.sha256(folder.encode())
    FILENAME_CACHE = hash_obj.hexdigest()
    if recursive:
        FILENAME_CACHE += '-R'
    FILENAME_CACHE += ext + '.bin'

    if cache:
        try:
            with open(FILENAME_CACHE, 'rb') as f:
                return pickle.load(f)
        except IOError:
            pass

    filenames = []
    if recursive:
        for root, dummy, fnames in os.walk(folder):
            for fname in fnames:
                if fname.endswith(ext):
                    if full_path:
                        filenames.append(os.path.join(root, fname))
                    else:
                        filenames.append(fname)
    else:
        for fname in os.listdir(folder):
            if fname.endswith(ext):
                if full_path:
                    filenames.append(os.path.join(folder,fname))
                else:
                    filenames.append(fname)

    if cache:
        if not os.path.exists(FILENAME_CACHE_DIR):
            os.makedirs(FILENAME_CACHE_DIR)
        with open(FILENAME_CACHE, 'wb') as f:
            pickle.dump(filenames, f, True)

    return filenames


def makedir_if_not_exist(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)  
    except FileExistsError:
        pass
    return directory


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


class StoreDictKeyPair(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(StoreDictKeyPair, self).__init__(*args, **kwargs)
        self.kv_dict = {}
    def __call__(self, parser, namespace, values, option_string=None):
        for kv in values.split(","):
            k, v = kv.split("=")
            if v.upper() == 'TRUE':
                self.kv_dict[k] = True
            elif v.upper() == 'FALSE':
                self.kv_dict[k] = False
            elif v.isdigit():
                self.kv_dict[k] = int(v)
            elif isfloat(v):
                self.kv_dict[k] = float(v)
            else:
                self.kv_dict[k] = v
        setattr(namespace, self.dest, self.kv_dict)
