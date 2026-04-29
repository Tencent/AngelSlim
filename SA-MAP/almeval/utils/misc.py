# flake8: noqa: F401, F403
import csv
import hashlib
import json
import mimetypes
import os
import os.path as osp
import pickle
import subprocess

import numpy as np
import pandas as pd
import validators
from loguru import logger


def download_file(url, filename=None):
    import urllib.request

    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if filename is None:
        filename = url.split('/')[-1]

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(
                url, filename=filename, reporthook=t.update_to)
    except Exception:
        # Handle Failed Downloads from huggingface.co
        if 'huggingface.co' in url:
            url_new = url.replace('huggingface.co', 'hf-mirror.com')
            try:
                os.system(f'wget {url_new} -O {filename}')
            except Exception:
                raise Exception(f'Failed to download {url}')
        else:
            raise Exception(f'Failed to download {url}')

    return filename


AUDIO_TYPES = {'mp3', 'ogg', 'wav', 'flac', 'm4a', 'wma', 'aac'}


def md5(s):
    hash = hashlib.new('md5')
    if osp.exists(s):
        with open(s, 'rb') as f:
            for chunk in iter(lambda: f.read(2**20), b''):
                hash.update(chunk)
    else:
        hash.update(s.encode('utf-8'))
    return str(hash.hexdigest())


def parse_file(s):
    if isinstance(s, str) and osp.exists(s) and s != '.':
        assert osp.isfile(s)
        suffix = osp.splitext(s)[1].lower()
        if suffix in AUDIO_TYPES:
            mime = 'audio'
        else:
            mime = mimetypes.types_map.get(suffix, 'unknown')
        return (mime, s)
    elif validators.url(s):
        suffix = osp.splitext(s)[1].lower()
        if suffix in AUDIO_TYPES:
            mime = 'audio'
        elif suffix in mimetypes.types_map:
            mime = mimetypes.types_map[suffix]
            
        return (mime, s)
    else:
        return (None, s)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4,
                  ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder)
                 for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_urls': False,
                                                                                      'strings_to_formulas': False}})

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False,
                    encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl,
                    xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    handlers = dict(pkl=load_pkl, json=load_json,
                    jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def run_command(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd).decode()


def print_once(msg):
    if not hasattr(print_once, 'printed'):
        print_once.printed = set()
    if msg not in print_once.printed:
        print_once.printed.add(msg)
        logger.info(msg)
