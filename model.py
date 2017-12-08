import pandas as pd
from transform import load_transform
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def feature_eng(df):
	'''
	'''

	# trim rows with 0 MB of traffic & extraneous data
	df = df[df['TotalTraffic_MB'] != 0]
	df = df[df['AP_Radio'] != '802.11a'] # drop single row with 802.11a

	# device features
	df['Apple'] = df['Vendor'].apply(lambda x: 1 if x == 'Apple, Inc.' else 0)
	radio_dummies = pd.get_dummies(df['AP_Radio'], prefix='radio')
	df = pd.concat([df, radio_dummies], axis=1)

	# time of day features
	df['work_hours'] = df['ConnectTime'].apply(lambda x: 1 if x.hour in np.arange(8, 17) else 0)

	# duration features (not used in model)
	# should be in minutes?
	df['Short'] = df['Duration'].apply(lambda x: 1 if x.seconds == 600 else 0)
	df['seconds'] = df['Duration'].apply(lambda x: x.seconds / 60)
	df['Long'] = df['seconds'].apply(lambda x: 1 if x > 40 else 0)

	# location featues
	wifi_dummies = pd.get_dummies(df['WiFiGroup'], prefix='at')
	df = pd.concat([df, wifi_dummies], axis=1)
	df['at_other'] = df['WiFiGroup'].apply(lambda x: 1 if x not in ['mainlib', 'ebcc', 'nbrc', 'psb', 'Civic Area Outside Wi-Fi East', 'sbrc'] else 0)

	# power user featues (trying to predict! 'y' series)
	df['high_power'] = df['TotalTraffic_MB'].apply(lambda x: 1 if x > 100 else 0)

	return df

if __name__ == '__main__':
	# df = load_transform()
	# df = feature_eng(df)

	X = df[[# decive features
			'Apple'
		   ,'radio_802.11ac'
		   ,'radio_802.11an'
		   ,'radio_802.11bgn'
		    # usage featues
		   ,'AvgUsage_Kbps'
		   ,'AvgSignal_dBm'
		   ,'AvgSignalQuality'
		    # time of day features
		   ,'work_hours'
		    # location features
		   ,'at_Civic Area Outside Wi-Fi East'
		   ,'at_ebcc'
		   ,'at_mainlib'
		   ,'at_nbrc'
		   ,'at_psb'
		   ,'at_sbrc'
		   ,'at_other'
 	   	   ]]

	y = df[['high_power']]

	X_train, X_test, y_train, y_test = train_test_split(X, y)

	rfc = RandomForestClassifier(n_estimators=50)
	rfc.fit(X_train, np.ravel(y_train))
	y_pred = rfc.predict(X_test)
	score = rfc.score(X_test, y_test)
	prfs = precision_recall_fscore_support(y_test, y_pred)
	cm = confusion_matrix(y_test, y_pred)
