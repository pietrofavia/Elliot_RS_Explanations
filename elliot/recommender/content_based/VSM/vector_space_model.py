"""
Module description:

"""


__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pickle
import time
import typing as t
import scipy.sparse as sp
import pandas as pd
import math
import csv

from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.content_based.VSM.vector_space_model_similarity import Similarity
from elliot.recommender.content_based.VSM.tfidf_utils import TFIDF
from elliot.recommender.base_recommender_model import init_charger

from os import path
from elliot.dataset.dataset import DataSet
from elliot.dataset.dataset import DataSetLoader
from elliot.namespace.namespace_model_builder import NameSpaceBuilder

class VSM(RecMixin, BaseRecommenderModel):
    r"""
    Vector Space Model

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/2362499.2362501>`_ and the `paper <https://ieeexplore.ieee.org/document/9143460>`_

    Args:
        similarity: Similarity metric
        user_profile:
        item_profile:

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        VSM:
          meta:
            save_recs: True
          similarity: cosine
          user_profile: binary
          item_profile: binary
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_user_profile_type", "user_profile", "up", "tfidf", None, None),
            ("_item_profile_type", "item_profile", "ip", "tfidf", None, None),
            ("_loader", "loader", "load", "ItemAttributes", None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        if self._user_profile_type == "tfidf":
            self._tfidf_obj = TFIDF(self._side.feature_map)
            self._tfidf = self._tfidf_obj.tfidf()
            self._user_profiles = self._tfidf_obj.get_profiles(self._ratings)
        else:
            self._user_profiles = {user: self.compute_binary_profile(user_items)
                                   for user, user_items in self._ratings.items()}

        self._i_user_feature_dict = {self._data.public_users[user]: {self._side.public_features[feature]: value
                                                                     for feature, value in user_features.items()}
                                     for user, user_features in self._user_profiles.items()}
        self._sp_i_user_features = self.build_feature_sparse_values(self._i_user_feature_dict, self._num_users)

        if self._item_profile_type == "tfidf":
            self._tfidf_obj = TFIDF(self._side.feature_map)
            self._tfidf = self._tfidf_obj.tfidf()
            self._i_item_feature_dict = {
                i_item: {self._side.public_features[feature]: self._tfidf[item].get(feature, 0)
                         for feature in self._side.feature_map[item]}
                for item, i_item in self._data.public_items.items()}
            self._sp_i_item_features = self.build_feature_sparse_values(self._i_item_feature_dict, self._num_items)
        else:
            self._i_item_feature_dict = {i_item: [self._side.public_features[feature] for feature
                                                  in self._side.feature_map[item]]
                                         for item, i_item in self._data.public_items.items()}
            self._sp_i_item_features = self.build_feature_sparse(self._i_item_feature_dict, self._num_items)

        self._model = Similarity(self._data, self._sp_i_user_features, self._sp_i_item_features, self._similarity)

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in self._ratings.keys()}

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    @property
    def name(self):
        return f"VSM_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start = time.time()
        self._model.initialize()
        end = time.time()
        self.logger.info(f"The similarity computation has taken: {end - start}")
        # recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        self.recs_val, self.recs_test = self.process_protocol(self.evaluator.get_needed_recommendations())

        self.pointwise_explanations()
        self.pairwise_explanations()
        # self.single_pairwise_explanations()


        self.evaluate()

    def compute_binary_profile(self, user_items_dict: t.Dict):
        user_features = {}
        # partial = 1/len(user_items_dict)
        for item in user_items_dict.keys():
            for feature in self._side.feature_map.get(item, []):
                # user_features[feature] = user_features.get(feature, 0) + partial
                user_features[feature] = user_features.get(feature, 1)
        return user_features

    def build_feature_sparse(self, feature_dict, num_entities):

        rows_cols = [(i, f) for i, features in feature_dict.items() for f in features]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(num_entities, len(self._side.public_features)))
        return data

    def build_feature_sparse_values(self, feature_dict, num_entities):
        rows_cols_values = [(u, f, v) for u, features in feature_dict.items() for f, v in features.items()]
        rows = [u for u, _, _ in rows_cols_values]
        cols = [i for _, i, _ in rows_cols_values]
        values = [r for _, _, r in rows_cols_values]

        data = sp.csr_matrix((values, (rows, cols)), dtype='float32',
                             shape=(num_entities, len(self._side.public_features)))

        return data

    def pointwise_explanations(self):

        print('\nstart pointwise\n')

        private_items = self._data.private_items
        public_items = self._data.public_items
        private_users = self._data.private_users
        public_users = self._data.public_users
        feature_map = self._side.feature_map #{private_item_id : [private_features_id]}
        private_features = self._side.private_features
        public_features = self._side.public_features

        feature_names = self.load_feature_names("data/cat_dbpedia_movielens_small/features2.tsv")

        pass

        """ MULTI USER MULTI RECS INDEX POINTWISE EXPLANATION """
        print('\nstart matching recs_item_features with user_profile_features\n')
        user_id = 1
        i = 0  # recs_index = 0->9
        # feature_recs_usprof_df = pd.DataFrame(columns=["user_id","recs_index","item_id","item_desc","feature_id","feature_desc"])
        # self._i_item_feature_dict[public_items[self.recs_test[1][0][0]]] <- le features del 1°item raccomandato dell'utente 1

        pointwise_df=pd.DataFrame(columns=["recs"])

        while user_id < len(public_users): #len(public_users),user_id in recs_test goes from 1 to 610
            while i < len(self.recs_test[user_id]): #len(self.recs_test[user_id]):
                for k in self._i_item_feature_dict[public_items[self.recs_test[user_id][i][0]]].keys():
                    if k in private_features and private_features[k] in self._user_profiles[user_id].keys():

                        new_row = {
                            'recs':['Recommendation n° ' + str(i) + ' To user ' + str(user_id) + ' was recommended item ' \
                                     + str(self.searchitem(self.recs_test[user_id][i][0])) \
                                     + str(self.searchfeat(private_features[k]))]
                        }
                        pointwise_df.loc[len(pointwise_df)] = new_row

                        print('row added',user_id,i)
                i = i + 1
            user_id = user_id + 1
            i=0

        print(pointwise_df.loc[0:len(pointwise_df)]['recs'])
        pointwise_df.to_csv('data/cat_dbpedia_movielens_small/pointwise_explanations0.tsv', sep='\t', header=None, index=False)

        with open("data/cat_dbpedia_movielens_small/pointwise_explanations0.tsv") as f_in:
            with open("data/cat_dbpedia_movielens_small/pointwise_explanations.tsv", "a", encoding="utf-8") as f_out:
                for line in f_in.readlines():
                    f_out.write(line.replace("[","").replace("]", "").replace("\"", "").replace("'", "").replace("Â",""))

        print('\nend pointwise\n')

    def single_pairwise_explanations(self,user_id : int = 1,recs_ind1 : int = 0,recs_ind2 : int = 1):
        print('\nstart pairwise\n')
        private_features = self._side.private_features
        public_features = self._side.public_features
        public_items = self._data.public_items
        line=[]
        # [feature_names[private_features[feature]] for feature in self._i_item_feature_dict[88].keys()]

        """ SINGLE USER TWO RECS PAIRWISE EXPLANATION
            Takes 2 recs of a specific user, compares them 
            and shows the features for which one rec is better than another"""

        if recs_ind1>recs_ind2:
            temp = recs_ind1
            recs_ind1 = recs_ind2
            recs_ind2 = temp

        recs_item1 = self.recs_test[user_id][recs_ind1][0] #1° recommended public item
        recs_item2 = self.recs_test[user_id][recs_ind2][0] #2° recommended public item

        #in recs test le recs sono in ordine decrescente dal max_tfidf_value in giù

        #in user_profiles={public_user_id:{public_feature_id:tfidf_value}} abbiamo il ranking delle features per tfidf_value
        print('searching best_feature of recs_item1 > best_feature of recs_item2...\n')

        # private features of recs_items
        first_feat_priv = self._i_item_feature_dict[public_items[recs_item1]]
        second_feat_priv = self._i_item_feature_dict[public_items[recs_item2]]

        # public features of recs_items
        first_feat_pub = {private_features[k]: v for k, v in first_feat_priv.items()}
        second_feat_pub = {private_features[k]: v for k, v in second_feat_priv.items()}

        #intersection : Given 2 dictionaries, the task is to find the intersection of these two dictionaries through keys.
        intersec_first_feat_k = set(self._user_profiles[user_id].keys()).intersection(set(first_feat_pub.keys()))
        intersec_second_feat_k = set(self._user_profiles[user_id].keys()).intersection(set(second_feat_pub.keys()))

        first_feat_ordpub = self.order_dict({k: self._user_profiles[user_id][k] * first_feat_pub[k] for k in intersec_first_feat_k})
        second_feat_ordpub = self.order_dict({k: self._user_profiles[user_id][k] * second_feat_pub[k] for k in intersec_second_feat_k})
        # best_item1_featpub = list(first_feat_ordpub.keys())[0]
        # best_item2_featpub = list(second_feat_ordpub.keys())[0]

        # PRINT OUT INDEXES ON CONSOLE
        # print('Recommendation n°',recs_ind1,'advising',recs_item1 ,'is more suitable for user',user_id,
        #       'than recommendation n°',recs_ind2,'advising',recs_item2,', because feature n°',best_item1_featpub,
        #       'is preferred over feature n°',best_item2_featpub)

        # PRINT OUT ON CONSOLE
        # print('Recommendation n°', recs_ind1,'advising',str(self.searchitem(recs_item1)) ,
        #       'is more suitable for user', user_id,'than recommendation n°', recs_ind2,
        #       'advising',str(self.searchitem(recs_item2)) ,', because feature',
        #       str(self.searchfeat2(list(first_feat_ordpub.keys()))),'is preferred over feature',
        #       str(self.searchfeat2(list(second_feat_ordpub.keys()))))

        # WRITE A TSV FILE
        line = ['Recommendation n° '+str(recs_ind1)+' advising '+str(self.searchitem(recs_item1))+\
               ' is more suitable for user '+str(user_id)+' than recommendation n° '+str(recs_ind2)+\
               ' advising '+str(self.searchitem(recs_item2))+', because feature '+str(self.searchfeat2(list(first_feat_ordpub.keys())))+\
               ' is preferred over feature '+str(self.searchfeat2(list(second_feat_ordpub.keys())))]

        print('\nstart writing... \n')
        fdf = pd.DataFrame.from_dict(line)
        fdf.to_csv('data/cat_dbpedia_movielens_small/pairwise_explanations.tsv',mode='a', sep="\t", header=None, index=False)
        print('done.')

        print('\nend pairwise\n')

    def pairwise_explanations(self):
        """MULTI USER MULTI RECS PAIRWISE EXPLANATION"""
        user_id = 1
        recs_ind1 = input("\ninsert first recs id to compare : ")
        recs_ind2 = input("\ninsert second recs id to compare : ")
        if recs_ind1 == '' or recs_ind2 == '':
            for i in range(0,len(self.recs_test)):
              self.single_pairwise_explanations(user_id)
              user_id = user_id + 1
        else :
            for i in range(0,len(self.recs_test)):
              self.single_pairwise_explanations(user_id,int(recs_ind1),int(recs_ind2))
              user_id = user_id + 1

    def load_feature_names(self, infile, separator='\t'):
        feature_names = {}
        with open(infile, encoding="latin-1") as file:
            for line in file:
                line = line.split(separator)
                pattern = line[1].split('><')
                pattern[0] = pattern[0][1:]
                pattern[len(pattern) - 1] = pattern[len(pattern) - 1][:-2]
                feature_names[int(line[0])] = pattern
        return feature_names

    def order_dict(self, dict_):
        return {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}

    def searchitem(self,i):
        print('\nfinding item...\n')  # in movies
        item_df = pd.read_csv("data/cat_dbpedia_movielens_small/movies.tsv", sep='\t', header=None)
        for index, row in item_df.iterrows():
            if i == row[0]:
                item_desc = row[1]
        return item_desc

    def searchfeat(self,f):
        print('\nfinding feature...\n')  # in feature_names = self.load_feature_names("data/cat_dbpedia_movielens_small/features2.tsv")
        feature_names = self.load_feature_names("data/cat_dbpedia_movielens_small/features2.tsv")
        for k,v in feature_names.items():
            if f == k :
                feature_desc = v
                break

        passlist = []
        passlist = feature_desc
        passlist[0] = self.cleantext(passlist[0])
        passlist[1] = self.cleantext(passlist[1])
        if passlist[0]=='':
            feature_desc = ' because there is this feature : ' + passlist[1]
        else:
            feature_desc = ' because there are these features : '+ passlist[0]+","+passlist[1]

        return feature_desc

    def searchfeat2(self,f):
        print('\nfinding feature...\n')  # in feature_names = self.load_feature_names("data/cat_dbpedia_movielens_small/features2.tsv")
        # [feature_names[private_features[feature]] for feature in self._i_item_feature_dict[88].keys()]
        feature_names = self.load_feature_names("data/cat_dbpedia_movielens_small/features2.tsv")

        i = 0
        name = []
        while name == []:
            if f[i] in feature_names:
                name = feature_names[f[i]]
            else:
                i = i + 1
        # name1 = self.cleantext(name[0]) , first link doesn't pass cleantext() function
        name2 = self.cleantext(name[1])

        return name2

    def cleantext(self,feature_desc):
        print('\nstart cleaning text ...\n')
        feat_cleanlist = ['http://', 'dbpedia.org', 'ontology', 'wikiPageWikiLink'
            , '/resource/','/page/','purl.org/','dc/','terms/','subject','gl.']
        for i in range(0, len(feat_cleanlist)):
            feature_desc = feature_desc.replace(feat_cleanlist[i], '')
        return feature_desc
