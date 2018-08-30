import data_process as dp
from sklearn.linear_model import LogisticRegression as lg
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score


# # C = 1e5
# mean accuracy is :
# 0.8911623605571932
# roc is :
# 0.6654517561630889

# # C = 1e4
# mean accuracy is :
# 0.8911623605571932
# roc is :
# 0.6654558531420531

# # C = 1000
# mean accuracy is :
# 0.8912201606843535
# roc is :
# 0.6654734562635344

# # C = 100
# mean accuracy is :
# 0.8912201606843535
# roc is :
# 0.6655491462138879

# # C = 10
# mean accuracy is :
# 0.8913935610658343
# roc is :
# 0.6665716896670371

# # C = 1
# mean accuracy is :
# 0.8916247615744755
# roc is :
# 0.6694786351227062

# # C = 0.1
# mean accuracy is :
# 0.8917981619559563
# roc is :
# 0.6744560479221546

# # C = 0.01
# mean accuracy is :
# 0.8918559620831166
# roc is :
# 0.6718558550863821

# # C = 0.001
# mean accuracy is :
# 0.892144962718918
# roc is :
# 0.6548395390412584


def train():
	print("load data ...")
	X, y = dp.onehot_process()
	print("finish loading data")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, random_state=1)

	np.save("X_train.npy", X_train)
	np.save("X_test.npy", X_test)
	np.save("y_train.npy", y_train)
	np.save("y_test.npy", y_test)

	model = lg(C=1e-3)
	print("train model ...")
	model.fit(X_train, y_train)
	print("finish training")

	# store the model
	joblib.dump(model, 'logistic_model.sav')


	
def predict():
	# X_train = np.load("X_train.npy")
	# y_train = np.load("y_train.npy")
	X_test = np.load("X_test.npy")
	y_test = np.load("y_test.npy")

	# print(X_train.shape)
	# print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)

	# load model
	model = joblib.load('logistic_model.sav')
	
	print("mean accuracy is : ")
	print(model.score(X_test, y_test))
	print("roc is : ")
	print(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

train()
predict()

