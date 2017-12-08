import pandas as pd
from transform import load_transform
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
	# df = load_transform()
	# df = df[df['TotalTraffic_MB'] != 0]
	# df['Apple'] = df['Vendor'].apply(lambda x: 1 if x == 'Apple, Inc.' else 0)
	# df['Short'] = df['Duration'].apply(lambda x: 1 if x.seconds == 600 else 0)
	# df['seconds'] = df['Duration'].apply(lambda x: x.seconds / 60)
	df['Long'] = df['seconds'].apply(lambda x: 1 if x > 40 else 0)
	# df['high_power'] = df['TotalTraffic_MB'].apply(lambda x: 1 if x > 3.38 else 0)
	# wifi_dummies = pd.get_dummies(df['WiFiGroup'], prefix='at')
	# df = pd.concat([df, wifi_dummies], axis=1)

	X = df[['Apple'
		   ,'AvgUsage_Kbps'
		   ,'AvgSignal_dBm'
		   ,'AvgSignalQuality'
		   # ,'TotalTraffic_MB' #colinear!!
		   # ,'high_power' #colinear!!
		   ,'at_1301'
		   ,'at_172014th'
		   ,'at_Civic Area Outside Wi-Fi East'
		   ,'at_Civic Area Outside Wi-Fi West'
		   ,'at_airport'
		   ,'at_atrium'
		   ,'at_bpl-north'
		   ,'at_carnegie'
		   ,'at_cg'
		   ,'at_cherry-n'
		   ,'at_cherry-s'
		   ,'at_cyf'
		   ,'at_ebcc'
		   ,'at_golf'
		   ,'at_grb'
		   ,'at_hill'
		   ,'at_iris'
		   ,'at_jc'
		   ,'at_mainlib'
		   ,'at_meadows'
		   ,'at_muni'
		   ,'at_nb'
		   ,'at_nbrc'
		   ,'at_osmpannex'
		   ,'at_osmpmaint'
		   ,'at_parkops'
		   ,'at_pc'
		   ,'at_pearl'
		   ,'at_psb'
		   ,'at_ranger'
		   ,'at_res'
		   ,'at_sbrc'
		   ,'at_scp'
	       ,'at_valmont'
		   ,'at_wsenior'
 	   	   ]]

	y = df[['Long']]

	X_train, X_test, y_train, y_test = train_test_split(X, y)

	rfc = RandomForestClassifier()
	rfc.fit(X_train, np.ravel(y_train))
	y_pred = rfc.predict(X_test)
	score = rfc.score(X_test, y_test)
	print score

	# clf = linear_model.SGDRegressor()
	# lso = linear_model.Lasso()
	# lso.fit(X_train, np.ravel(y_train))
	#
	# y_pred = lso.predict(X_test)
	# score = lso.score(X_test, y_test)
