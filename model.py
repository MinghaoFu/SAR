import os
import sys
import numpy as np
import pandas as pd
import json
import logging
from scipy import sparse
from sklearn.utils import shuffle
from tqdm import tqdm
from IPython.display import display
import pyspark

from sklearn.preprocessing import MinMaxScaler

from recommenders.utils.python_utils import (
    jaccard,
    get_top_k_scored_items,
    rescale,
)

from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation
from recommenders.evaluation.python_evaluation import auc, logloss

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

COOCCUR = "cooccurrence"
JACCARD = "jaccard"
LIFT = "lift"

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

        # Train dataset
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

        # Valid dataset
        self.behaviors_dev = pd.read_csv(self.valid_behaviors_file, sep='\t', names=['Impression ID',
                                                                                     'User ID',
                                                                                     'Time',
                                                                                     'History',
                                                                                     'Impressions'])

        print(f'Valid behaviors length: {len(self.behaviors_dev)}')


class MINDmodel:
    def __init__(
            self,
            data,
            col_user='User ID',
            col_item='News ID',
            col_time='Time',
            col_rating='Rating',
            col_history='History',
            col_impression='Impression ID',
            col_prediction='Prediction',
            time_decay_flag=True,
            similarity_type=JACCARD,
            threshold=1,
            time_now=None
    ):
        self.col_user = col_user
        self.col_item = col_item
        self.col_time = col_time
        self.col_rating = col_rating
        self.col_history = col_history
        self.col_impression = col_impression
        self.col_timedecay = None
        self.similarity_type = similarity_type
        self.threshold = threshold
        self.time_now = time_now
        self.time_decay_flag = time_decay_flag
        self.col_prediction = col_prediction

        self.n_users = None
        self.n_items = None

        self.user2index = None
        self.item2index = None

        self.index2item = None
        self.index2user = None

        self.rating_df = None
        self.user_item_time_dict = None
        self.click_df = None
        self.wiki_dict = None

        self.affinity_matrix = None

        self.user_coocurrence = None
        self.u2u_sim = None

        self.item_coocurrence = None
        self.i2i_sim = None

        self.item_embedding_dict = None

    def get_recent_behaviors_dict(self):
        user_hist_imp = {}
        for i, row in tqdm(data.behaviors.iterrows()):
            if user_hist_imp.get(row['User ID']) is None:
                user_hist_imp[row['User ID']] = str(row['History']).split(' '), str(row['Impressions']).split(
                    ' '), pd.to_datetime(row['Time'])
            else:
                if pd.to_datetime(row['Time']) > user_hist_imp[row['User ID']][2]:
                    user_hist_imp[row['User ID']] = str(row['History']).split(' '), user_hist_imp[row['User ID']][
                        1] + str(row['Impressions']).split(' '), pd.to_datetime(row['Time'])
                else:
                    user_hist_imp[row['User ID']] = user_hist_imp[row['User ID']][0], user_hist_imp[row['User ID']][
                        1] + str(row['Impressions']).split(' '), user_hist_imp[row['User ID']][2]

        data.behaviors = user_hist_imp

    def set_index(self, behaviors_dict):
        '''
        Mapping user and item id to index
        :param behaviors_df: User impression history dataframe
        :param news_df: news dataframe
        :return: None
        '''
        logger.info('Create id2index mapping...')

        self.index2user = dict(enumerate(behaviors_dict.keys()))
        self.user2index = {v: k for k, v in self.index2user.items()}

        self.index2item = dict(enumerate(data.news[self.col_item]))
        self.item2index = {v: k for k, v in self.index2item.items()}

        self.n_users = len(self.index2user)
        self.n_items = len(self.index2item)

        print('User num: {}'.format(self.n_users))
        print('Item num: {}'.format(self.n_items))

    def get_rating_df(self, behaviors):
        '''
        Get all click history with rating and timedecay
        :param behaviors: User impressions history dataframe
        :return: None
        '''

        def set_time_decay(df, col_timedecay, half_life):
            '''
            Calculate click actions timedecay
            :param df: behaviors dataframe
            :param col_timed: name of time column in df
            :param half_life: hyper parameter T
            :return: None
            '''
            logger.info('get time decay of rating dataframe...')

            self.col_timedecay = col_timedecay
            if self.time_now is None:
                self.time_now = pd.to_datetime(df[self.col_time]).max()
            print(f'time now: {self.time_now}')

            df[self.col_timedecay] = df.apply(
                lambda x: pd.to_numeric(x[self.col_rating]) * np.power(0.5, (
                        pd.to_datetime(self.time_now) - pd.to_datetime(x[self.col_time])).days / half_life),
                axis=1
            )

        logger.info('Create rating dataframe...')

        rating_df_path = os.path.join('data', 'train', 'rating.csv')
        if not os.path.exists(rating_df_path):
            lst = []
            for k, v in tqdm(behaviors.items()):
                for hist in v[0]:
                    if self.item2index.get(hist) is not None:
                        lst.append([self.user2index[k], self.item2index[hist], 2])
                for imp in v[1]:
                    if imp[-1] == '1':
                        lst.append([self.user2index[k], self.item2index[imp[:-2]], 3])
                    else:
                        lst.append([self.user2index[k], self.item2index[imp[:-2]], 1])

            self.rating_df = pd.DataFrame(lst, columns=[self.col_user, self.col_item, self.col_rating])
            # timedecay
            self.rating_df.to_csv(rating_df_path, index=False)

            logger.info(f'Rating dataframe has been saved as {rating_df_path}')

        self.col_timedecay = 'Timedecay'

        logger.info(f'Read rating dataframe from {rating_df_path}')
        self.rating_df = pd.read_csv(rating_df_path, names=
        [self.col_user, self.col_item, self.col_rating]).iloc[1:]
        self.rating_df = self.rating_df.astype(
            {self.col_user: 'int64', self.col_item: 'int64', self.col_rating: 'float64'})

    #         if self.time_decay_flag is True:
    #             self.rating_df[self.col_rating] *= self.rating_df[self.col_timedecay]

    def get_click_df(self, behaviors):
        '''
        Create id to number dictionary
        :param behaviors: dataframe
        :return: None
        '''
        logger.info('Create click dataframe...')

        click_df_path = os.path.join('data', 'train', 'click.csv')
        if not os.path.exists(click_df_path):
            lst = []
            for i, row in tqdm(behaviors.iterrows()):
                imp_lst = row['Impressions'].split(' ')
                for imp in imp_lst:
                    if imp[-1] == '1':
                        lst.append([self.user2index[row['User ID']], self.item2index[imp[:-2]], row[self.col_time]])

            self.click_df = pd.DataFrame(lst, columns=[self.col_user, self.col_item, self.col_time])
            self.click_df.to_csv(click_df_path, index=False)
            logger.info(f'Click dataframe has been saved as {click_df_path}')

        logger.info(f'Read click dataframe from {click_df_path}')
        self.click_df = pd.read_csv(click_df_path, names=[self.col_user, self.col_item, self.col_time]).iloc[1:]

    def get_item_topk_click(self, click_df, k):
        return click_df[self.col_item].value_counts().index[:k]

    def get_hist_and_last_clicks(self, click_df):
        '''
        Get user the last click and other click history
        :param click_df: click history
        :return:
        click_hist_df: all users click history except the last one
        click_last_df: all users last click
        '''
        click_df = click_df.sort_values(by=[self.col_user, self.col_time])
        click_last_df = click_df.groupby(self.col_user).tail(1)

        def hist_func(user_df):
            if len(user_df) == 1:
                return user_df
            else:
                return user_df[:-1]

        click_hist_df = click_df.groupby('User ID').apply(hist_func).reset_index(drop=True)

        return click_hist_df, click_last_df

    def get_user_item_time_dict(self, click_df):
        '''
        Create user-[(item1, time1), (item2, time2)...] dictionary
        :param click_df: click history dataframe
        :return: None
        '''
        logger.info('Create user-(item, time) dictionary dataframe...')

        click_df = click_df.sort_values('Time')

        def make_item_time_pair(df):
            return list(zip(df['News ID'], df['Time']))

        user_item_time_df = click_df.groupby('User ID')['News ID', 'Time'] \
            .apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'News-Time list'})

        self.user_item_time_dict = dict(zip(user_item_time_df['User ID'], user_item_time_df['News-Time list']))

    def get_wiki_dict(self, en_path, re_path):
        '''
        Create WikidataId-embedding_vector dictionary
        :param en_path: entity_embedding file
        :param re_path: relation_embedding file
        :return: None
        '''
        logger.info('Create news-wikiVector dictionary dataframe...')

        res = []
        with open(en_path) as f:
            for line in f:
                lst = line.split('\t')
                res.append([lst[0], lst[1:101]])

        with open(re_path) as f:
            for line in f:
                lst = line.split('\t')
                res.append([lst[0], lst[1:101]])

        tmp = pd.DataFrame(res)
        self.wiki_dict = dict(zip(tmp[0], tmp[1]))

    def get_item_embedding_dict(self, news, wiki_dict):
        '''
        Create item_embedingVector dictionary
        :param news: all items dataframe
        :param wiki_dict: WikidataId-embeddingVector dictionary
        :return: None
        '''
        logger.info('Create news-embedding dictionary...')

        def _str_find_all_prefix(str, sub):
            lst = []
            start = 0
            while True:
                start = str.find(sub, start)
                if start == -1:
                    return lst
                start += len(sub)
                lst.append(start)

        d_lst = []

        for info in news['Title Entities']:
            vec = [0.] * 100
            w_lst = []
            for a in _str_find_all_prefix(str(info), '\"WikidataId\": \"'):
                s = ''
                while info[a] != '"':
                    s += info[a]
                    a += 1
                if wiki_dict.get(s):
                    w_lst.append(s)

            for s in w_lst:
                for i, v in enumerate(wiki_dict[s]):
                    vec[i] += float(v)

            divisor = 1 if len(w_lst) == 0 else len(w_lst)
            d_lst.append([v / divisor for v in vec])

        self.item_embedding_dict = dict(zip(news[self.col_item], d_lst))

    def get_affinity_matrix(self, df):
        '''
        Calculate user-item affinity matrix
        :param df: rating/click dataframe
        :return: None
        '''
        logger.info('Create user affinity matrix...')

        self.affinity_matrix = sparse.coo_matrix(
            (pd.to_numeric(df[self.col_rating]), (df[self.col_user], df[self.col_item])),
            shape=(self.n_users, self.n_items),
        ).tocsr()

    def get_items_coocurrence_matrix(self, df):
        '''
        Calculate coocurence matrix for item-item similarity
        :param df: user-item impressions dataframe
        :return: None
        '''
        logger.info('Create items coocurrance matrix...')

        user_item_hits_mat = sparse.coo_matrix(
            (np.repeat(1, df.shape[0]), (df[self.col_user], df[self.col_item])),
            shape=(self.n_users, self.n_items),
        ).tocsr()

        self.item_cooccurrence = user_item_hits_mat.transpose().dot(user_item_hits_mat)
        self.item_cooccurrence = self.item_cooccurrence.multiply(
            self.item_cooccurrence >= self.threshold
        ).toarray()

    def get_item_similarity_matrix(self, type):

        def _jaccard(cooccurrence_mat):

            diag = cooccurrence_mat.diagonal()  # Items in cooccurence matrix must be found in behaviors, otherwise dignoal includes zero
            diag_rows = np.expand_dims(diag, axis=0)  # For broadcast
            diag_cols = np.expand_dims(diag, axis=1)

            with np.errstate(invalid="ignore", divide="ignore"):  # cij/(cii + cjj - cij)
                result = cooccurrence_mat / (diag_rows + diag_cols - cooccurrence_mat)

            return np.array(result)

        def _lift(cooccurrence_mat):

            diag = cooccurrence_mat.diagonal()
            diag_rows = np.expand_dims(diag, axis=0)
            diag_cols = np.expand_dims(diag, axis=1)

            with np.errstate(invalid="ignore", divide="ignore"):
                result = cooccurrence_mat / (diag_rows * diag_cols)

            return np.array(result)

        self.similarity_type = type
        logger.info('Create item similarity matrix...')
        if self.similarity_type == COOCCUR:
            logger.info('Using co-occurrence based similarity to build')
            self.i2i_sim = self.item_cooccurrence
        elif self.similarity_type == JACCARD:
            logger.info('Using jaccard based similarity to build')
            self.i2i_sim = jaccard(self.item_cooccurrence)
        elif self.similarity_type == LIFT:
            logger.info('Using lift based similarity to build')
            self.i2i_sim = _lift(self.item_cooccurrence)
        else:
            raise ValueError("Unknown similarity type: {self.similarity_type}")

    def get_user_coocurrence_matrix(self, df):
        '''
        Calculate coocurence matrix for user-user similarity
        :param df: user-item impressions dataframe
        :return: None
        '''
        logger.info('Create items coocurrance matrix...')

        user_item_hits_mat = sparse.coo_matrix(
            (np.repeat(1, df.shape[0]), (df[self.col_user], df[self.col_item])),
            shape=(self.n_users, self.n_items),
        ).tocsr()

        self.user_cooccurrence = user_item_hits_mat.dot(user_item_hits_mat.transpose())
        self.user_cooccurrence = self.user_cooccurrence.multiply(
            self.user_cooccurrence >= self.threshold
        ).toarray()

    def get_user_similarity_matrix(self, type):

        def _jaccard(cooccurrence_mat):

            diag = cooccurrence_mat.diagonal()  # Items in cooccurence matrix must be found in behaviors, otherwise dignoal includes zero
            diag_rows = np.expand_dims(diag, axis=0)  # For broadcast
            diag_cols = np.expand_dims(diag, axis=1)

            with np.errstate(invalid="ignore", divide="ignore"):  # cij/(cii + cjj - cij)
                result = cooccurrence_mat / (diag_rows + diag_cols - cooccurrence_mat)

            return np.array(result)

        def _lift(cooccurrence_mat):

            diag = cooccurrence_mat.diagonal()
            diag_rows = np.expand_dims(diag, axis=0)
            diag_cols = np.expand_dims(diag, axis=1)

            with np.errstate(invalid="ignore", divide="ignore"):
                result = cooccurrence_mat / (diag_rows * diag_cols)

            return np.array(result)

        self.similarity_type = type
        logger.info('Create item similarity matrix...')
        if self.similarity_type == COOCCUR:
            logger.info('Using co-occurrence based similarity to build')
            self.u2u_sim = self.item_cooccurrence
        elif self.similarity_type == JACCARD:
            logger.info('Using jaccard based similarity to build')
            self.u2u_sim = _jaccard(self.user_cooccurrence)
        elif self.similarity_type == LIFT:
            logger.info('Using lift based similarity to build')
            self.u2u_sim = _lift(self.user_cooccurrence)
        else:
            raise ValueError("Unknown similarity type: {self.similarity_type}")

    def get_user_activate_degree_dict(self, df):
        click_times_df = df.groupby(self.col_user)[self.col_item].count().reset_index()
        mm = MinMaxScaler()

        # normalization
        click_times_df[self.col_item] = mm.fit_transform(click_times_df[[self.col_item]])
        user_activate_degree_dict = dict(zip(click_times_df[self.col_user], click_times_df[self.col_item]))

        return user_activate_degree_dict

    def score_all_items(self, test):
        '''
        Score all items for test users
        :return: None
        '''
        logger.info('Calculate all items score...')

        user_ids = list(
            map(
                lambda user: self.user2index.get(user, np.NAN),
                test[self.col_user].unique()
            )  # mapping test user to index
        )
        if any(np.isnan(user_ids)):  # np.isnan return a ndarray
            raise ValueError('Model cannot score users that are not in train dataset')

        scores = self.affinity_matrix[user_ids, :].dot(self.i2i_sim)
        if isinstance(scores, sparse.spmatrix):
            scores = scores.toarray()

        return scores

    def user_items_score(self, user_id):
        return self.affinity_matrix[self.user2index[user_id]].dot(self.i2i_sim)

    def get_pop_topk(self, k, sort_topk: bool):
        '''
        Get top k popular items, according to item impressing times, which can be directly
        got from coocurrence matrix diagonal
        :param k: k items got
        :param sort_topk: whether sort top k items
        :return: item-prediction(aka score) dataframe
        '''
        logger.info('Calculate top {} popular items...'.format(k))
        imp_counts = self.item_cooccurrence.diagonal()

        # utils: score[test_user_idx, top_items] ?
        def topk_index_value(k, arr):
            if (k > len(arr)):
                logger.warning('k must be less than the array size')
                k = len(arr)

            top_items = sorted(range(len(arr)), key=lambda x: arr[x])[-k:]
            top_scores = [arr[i] for i in top_items]
            return np.array(top_items), np.array(top_scores)

        top_items, top_scores = topk_index_value(k, imp_counts)
        return pd.DataFrame(
            {
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

    def get_topk_score(self, k, scores, user_order, sort_topk):
        return sorted(range(len(data.news)), key=lambda x: scores[user_order][x])[-k:]


    def get_sim_item_topk(self, items, k, sort_topk: bool):
        '''
        Get top k items according to similarity matrix
        :param items: Input items
        :param k: number of items
        :param sort_topk: whether sort top k items
        :return: top k items
        '''
        # convert id to indices
        print(items)
        item_indices = np.asarray(
            list(
                map(
                    lambda x: self.item2index.get(x, np.NaN),
                    items
                )
            )
        )
        print(f'item_indices: {item_indices}')

        hist_item_num = len(self.i2i_sim)
        gen_sim = np.zeros((1, hist_item_num))
        for i in item_indices:
            gen_sim += self.i2i_sim[i]

        sorted_sim_item = sorted(range(hist_item_num), key=lambda x: gen_sim[x])
        return sorted_sim_item[-k:]

    def get_dev_userid_hist_dict(self, dev_df):

        logger.info('Create dev user-historyClick dictionary...')

        dev_userid_hist_dict = {}
        for i, row in tqdm(dev_df.iterrows()):
            if i == 0 or i == 1:
                print(row[self.col_history])
            user_click_hist = str(row[self.col_history]).split(' ')
            dev_userid_hist_dict[row[self.col_user]] = user_click_hist

        return dev_userid_hist_dict

    def fit(self):

        logger.info('Data prepare...')
        self.get_recent_behaviors_dict()
        self.set_index(data.behaviors)
        self.get_rating_df(data.behaviors)
        self.get_wiki_dict(data.entity_embedding_file, data.relation_embedding_file)
        self.get_item_embedding_dict(data.news, self.wiki_dict)
        self.get_affinity_matrix(self.rating_df)
        self.get_items_coocurrence_matrix(self.rating_df)
        self.get_item_similarity_matrix(COOCCUR)

        score_path = os.path.join('data','valid','score.npy')
        if not os.path.exists(score_path):
            scores = self.score_all_items(data.behaviors_dev)
            np.save(score_path, scores)

        scores = np.load(score_path)

        #pop_topk = self.get_pop_topk(200, False)
        score_topk = self.get_topk_score(40, scores, 0, True)

        i_scores = self.user_items_score(data.behaviors_dev[self.col_user].iloc[0])
        x = self.user2index['U41827']
        print([self.index2item[i] for i in self.affinity_matrix[x].toarray()])









    def predict(self, df):

        pop_topk = self.get_pop_topk(200, False)
        user_hist_dict = self.get_dev_userid_hist_dict(df)
        for i, row in tqdm(df.iterrows()):
            if self.user2index.get(row[self.col_user]) is not None:
                sim_topk = model.get_sim_item_topk(user_hist_dict[row[self.col_user]], 5, False)
                print(sim_topk)
                break


if __name__ == '__main__':
    data = MINDdata()
    model = MINDmodel(data)
    model.fit()
#     model.predict(data.behaviors_dev)
#     spark = start_or_get_spark()
#     rating_true = spark.createDataFrame(model.rating_df)



