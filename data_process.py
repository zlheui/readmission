import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder


def load_data():
	csv_files = []
	for file in os.listdir('..'):
		if '.csv' in file and 'all' not in file:
			csv_files.append(file)

	# print(len(csv_files))
	# print(csv_files)

	df_join = pd.read_csv('../'+csv_files[0])

	for i in range(1, len(csv_files)):
		df_join = pd.merge(df_join, pd.read_csv('../'+csv_files[i]), on=['encounter_id', 'patient_nbr'], validate='one_to_one')

	return df_join
	


def onehot_process():
	df = load_data()

	nb_row, _ = df.shape

	# print(type(df.columns.values))
	# print(df.columns.values)
	# print(df.dtypes)

	# print(df)
	# print(type(df.at[0, 'age']))
	# print(df.at[0, 'age'])


	# process integer type feature
	count_row = 0
	for _, row in df.iterrows():	
		for e in df.columns.values:
			if e == 'time_in_hospital':
				if row[e] < 1:
					df.at[count_row, e] = 0
				elif row[e] >= 4 and row[e] <= 6:
					df.at[count_row, e] = 4
				elif row[e] >=7 and row[e] <= 13:
					df.at[count_row, e] = 5
				else:
					df.at[count_row, e] = 6
			elif e == 'number_diagnoses':
				if row[e] <= 5:
					df.at[count_row, e] = 0
				elif row[e] >= 6 and row[e] <= 10:
					df.at[count_row, e] = 1
				elif row[e] >= 11 and row[e] <= 15:
					df.at[count_row, e] = 2
				else:
					df.at[count_row, e] = 3
			elif e == 'number_inpatient' or e == 'number_outpatient' or e == 'number_emergency':
				if row[e] >= 4:
					df.at[count_row, e] = 4
			elif e == 'num_lab_procedures' or e == 'num_medications':
				v_orig = df.at[count_row, e]
				v = v_orig // 10
				if v >= 10 and v % 10 == 0:
					v -= 1
				df.at[count_row, e] = v
		count_row += 1

	# X stores the covariate features
	X = np.array(df[['num_procedures', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'number_diagnoses', 'number_inpatient', 'number_outpatient', 'number_emergency', 'num_lab_procedures', 'num_medications']])


	# convert a string column into a category list
	def string_column_to_cateogry(col_name):
		df1 = df[col_name].tolist()
		tmp_dict = {}
		count_unique = 0
		for i in range(0, len(df1)):
			if df1[i] in tmp_dict.keys():
				df1[i] = tmp_dict[df1[i]]
			else:
				tmp_dict[df1[i]] = count_unique
				df1[i] = count_unique
				count_unique += 1
		return df1		


	# process string type features
	df1 = string_column_to_cateogry('medical_specialty')
	df2 = string_column_to_cateogry('race')
	df3 = string_column_to_cateogry('gender')
	df4 = df['age'].tolist()
	for i in range(0, len(df4)):
		if int(df4[i][1]) >= 6:
			df4[i] = 1
		else:
			df4[i] = 0

	# append string type features to X
	X = np.append(X, np.transpose(np.array([df1])), axis=1)
	X = np.append(X, np.transpose(np.array([df2])), axis=1)
	X = np.append(X, np.transpose(np.array([df3])), axis=1)
	X = np.append(X, np.transpose(np.array([df4])), axis=1)


	# convert categorical features in X to one hot encoding
	enc = OneHotEncoder()
	enc.fit(X)
	X = enc.transform(X).toarray()


	# process the three diagnosis features
	df5 = df['diag_1'].tolist()
	df6 = df['diag_2'].tolist()
	df7 = df['diag_3'].tolist()


	# only keep the first 3 digits of diagnosis code
	def transform_diagnosis_feature(col):

		for i in range(0, len(col)):
			if '.' in col[i]:
				col[i] = int(col[i].split('.')[0])
			elif 'V' in col[i] or 'E' in col[i]:
				col[i] = int(col[i][1:])
			elif '?' in col[i]:
				col[i] = 0
			else:
				col[i] = int(col[i])

		return col


	df5 = transform_diagnosis_feature(df5)
	df6 = transform_diagnosis_feature(df6)
	df7 = transform_diagnosis_feature(df7)

	tmp = np.append(np.array(df5), np.array(df6), axis=0)
	tmp = np.append(tmp, np.array(df7), axis=0)
	tmp = np.transpose(np.array([tmp]))

	enc = OneHotEncoder()
	enc.fit(tmp)
	tmp = enc.transform(tmp).toarray()

	df8 = np.logical_or(tmp[0:nb_row], tmp[nb_row:2*nb_row])
	df8 = np.logical_or(df8, tmp[2*nb_row:])


	X = np.append(X, df8, axis=1)


	# get the response variable
	Y = df['readmitted'].tolist()
	for i in range(0, len(Y)):
		if Y[i] == '<30':
			Y[i] = 1
		else:
			Y[i] = 0

	Y = np.array(Y)

	return X, Y

