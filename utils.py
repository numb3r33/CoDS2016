import pandas as pd


def load_data_set(train_filename='./data/train.csv', test_filename='./data/test.csv'):
	print 'Loading datasets'

	train = pd.read_excel('./data/train.xlsx', parse_dates=True)
	test = pd.read_excel('./data/test.xlsx', parse_dates=True)

	print 'Set ID as index'

	train = train.set_index('ID')
	test = test.set_index('ID')

	return train, test

def lowercase_column_names(train, test):
	# lowercase column names
	train.columns = train.columns.map(lambda x: x.lower())
	test.columns = test.columns.map(lambda x: x.lower())

	return train, test