import numpy as np

from sklearn.cross_validation import train_test_split, ShuffleSplit

def eval_models(clfs, X, y):
	cv = ShuffleSplit(X.shape[0], n_iterations=5, test_size=0.2, indices=True)

	scores = []

	for train_index, test_index in cv:
		preds_common = np.zeros((len(test_index), 2))

		for clf in clfs:
			X_train, y_train = X[train_index], y[train_index]
			X_test, y_test = X[test_index], y[test_index]

			clf.fit(X_train, y_train)

			preds = clf.predict(X_test)
			print  "score: %f" % np.mean((y_test - preds) ** 2)
			preds_common += preds

		preds_common /= 4.

		scores.append(np.mean((y_test - probs_common) ** 2))
		print "combined score: %f " %(scores[-1])

	print np.mean(scores), np.std(scores)