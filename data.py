import sys
import os
import numpy as np
import pandas as pd
import zipfile
from tqdm import tqdm
import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm
from collections import defaultdict

tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set



MIND_type = 'demo'
data_path = 'data'  # temp dir

class MINDdata:
    def __init__(self, MIND_type='demo', data_path='data'):
        self.train_news_file = os.path.join(data_path, 'train', r'news.tsv')
        self.train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
        self.valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
        self.valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
        self.entity_embedding_file = os.path.join(data_path, 'valid', r'entity_embedding.vec')
        self.relation_embedding_file = os.path.join(data_path, 'valid', r'relation_embedding.vec')

        mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

        if not os.path.exists(self.train_news_file):
            download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)

        if not os.path.exists(self.valid_news_file):
            download_deeprec_resources(mind_url, \
                                       os.path.join(data_path, 'valid'), mind_dev_dataset)

        self.behaviors = pd.read_csv(self.train_behaviors_file, sep='\t', names=['Impression ID',
                                                            'User ID',
                                                            'Time',
                                                            'History',
                                                            'Impressions'])

        self.news = pd.read_csv(self.train_news_file, sep='\t', names=['News ID',
                                                            'Category',
                                                            'SubCategory',
                                                            'Title',
                                                            'Abstract',
                                                            'URL',
                                                            'Title Entities',
                                                            'Abstract Entities'])
        print(f'Behaviors length: {len(self.behaviors)}')
        print(f'News length: {len(self.news)}')




