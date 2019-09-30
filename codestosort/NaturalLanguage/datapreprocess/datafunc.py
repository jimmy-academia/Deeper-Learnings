import os
import sys
sys.path.append('process')

from preprocessor import Preprocessor
from embedding import Embedding

import torch
import pickle
import logging
import json


def build_processed_data(datadir, pickledir, neg_num=9, last=False, difemb=False):
    if difemb:
        embedding_path = os.path.join(datadir,'wiki-news-300d-1M.vec')
    else:
        embedding_path = os.path.join(datadir,'crawl-300d-2M.vec')
    filenames = ['train', 'test', 'valid']
    filepaths = [os.path.join(datadir, '%s.json')%fn for fn in filenames]
    savepaths = [os.path.join(pickledir, '%s.pkl')%fn for fn in filenames]

    preprocessor = Preprocessor(None)

    # collect words appear in the data
    words = set()
    for path in filepaths:
        logging.info('collecting words from {}'.format(path))
        words |= preprocessor.collect_words(path, n_workers=4)

    # load embedding only for words in the data
    logging.info(
        'loading embedding from {}'.format(embedding_path)
    )
    embedding = Embedding(embedding_path, words)
    embedding_pkl_path = os.path.join(pickledir, 'embedding.pkl')
    logging.info('Saving embedding to {}'.format(embedding_pkl_path))
    with open(embedding_pkl_path, 'wb') as f:
        pickle.dump(embedding, f)

    # update embedding used by preprocessor
    preprocessor.embedding = embedding

    # make pkl for datasets
    for fn, filepath, savepath in zip(filenames, filepaths, savepaths):
        logging.info('Processing {} from {}'.format(fn, filepath))
        shuffle = True if fn =='train' else False
        n_positive = 1 if fn =='train' else -1
        n_negative = neg_num if fn =='train' else -1
        data = preprocessor.get_dataset(
            filepath, 4, {'n_positive': n_positive, 'n_negative': n_negative, 'shuffle': shuffle, 'last':last}
        )
        logging.info('Saving {} to {}'.format(fn, savepath))
        with open(savepath, 'wb') as f:
            pickle.dump(data, f)

 
def make_dataloader(outputdir, filenames = ['train', 'test', 'valid']):
    # read data
    dataloaders = []
    for fn in filenames:
        logging.info('loading {} dataset'.format(fn))
        data = get_datas(outputdir, fn)
        dataloader = torch.utils.data.DataLoader(data,
            batch_size=10, collate_fn=data.collate_fn)

        dataloaders.append(dataloader)
    return dataloaders

def get_datas(outputdir, fn='test'):
    filepath = os.path.join(outputdir, '%s.pkl'%fn)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data
