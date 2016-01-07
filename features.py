from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

import pandas as pd
import numpy as np

class FeatureTransformer(BaseEstimator):
	"""
	Encodes categorical features to numerical features
	"""

	def __init__(self, train, test):
		self.X = train
		self.X_test = test

	def get_feature_names(self):
		feature_names = []

		feature_names.extend(['year_of_birth'])
		feature_names.extend(self.categorical_features_columns)
		feature_names.extend(self.numerical_features_columns)

		return np.array(feature_names)

	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self

	def fit_transform(self, X, y=None):
		date_features = self._process_dates(X)
		
		# categorical_features = self._process_categorical_features(X)
		categorical_features = self._one_hot_encoded_features(X)
		numerical_features = self._process_numerical_features(X)

		features = []
		
		features.append(date_features)
		features.append(categorical_features)
		features.append(numerical_features)

		features = np.hstack(features)
		
		return features

	def _process_dates(self, X):
		'Only process date of birth for now'

		year_of_birth = X.dob.dt.year # returns year of birth

		return year_of_birth.reshape(-1, 1)

	def _process_categorical_features(self, X):
		'Encode categorical features into numerical features'

		self.categorical_features_columns = ['gender', 'degree', 'specialization', 'collegestate']
		categorical_features = []

		for cat in self.categorical_features_columns:
			lbl = LabelEncoder()

			lbl.fit(pd.concat([self.X[cat], self.X_test[cat]], axis=0))

			categorical_features.append(lbl.transform(X[cat]))

		return np.array(categorical_features).T

	def _one_hot_encoded_features(self, X):
		'One hot encoding of categorical variables'

		self.categorical_features_columns = ['gender', 'degree', 'specialization', 'collegestate']
		X_categorical = X[self.categorical_features_columns]

		data = pd.concat([self.X[self.categorical_features_columns], self.X_test[self.categorical_features_columns]], axis=0)
		one_hot_encoded_features = data.T.to_dict().values()

		vec = DictVectorizer()
		vec.fit(one_hot_encoded_features)

		encoded_features = vec.transform(X_categorical.T.to_dict().values())

		return encoded_features.toarray()


	def _process_numerical_features(self, X):
		'Return numerical features as it is'

		self.numerical_features_columns = ['10percentage', '12graduation', '12percentage', 'collegeid',
							  'collegetier', 'collegegpa', 'collegecityid', 'collegecitytier',
							  'graduationyear', 'english', 'logical', 'quant', 'domain', 'computerprogramming',
							  'electronicsandsemicon', 'computerscience', 'mechanicalengg', 'electricalengg',
							  'telecomengg', 'civilengg', 'conscientiousness', 'agreeableness', 'extraversion',
							  'nueroticism', 'openess_to_experience']

		numerical_features = []

		for col in self.numerical_features_columns:
			numerical_features.append(X[col])

		return np.array(numerical_features).T
		

	def transform(self, X):
		
		date_features = self._process_dates(X)
		# categorical_features = self._process_categorical_features(X)
		categorical_features = self._one_hot_encoded_features(X)
		numerical_features = self._process_numerical_features(X)

		features = []
		
		features.append(date_features)
		features.append(categorical_features)
		features.append(numerical_features)
		
		features = np.hstack(features)
		
		return features
	
