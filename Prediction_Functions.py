import joblib

def Connection(df2):

 conn=joblib.load('SaveModels/Model_Conn_English')
 prediction = conn.predict(df2['review'])
 print(prediction)
 df2['Connection'] = prediction
 return df2


def Vcsettings(df2):
 status = joblib.load('SaveModels/Model_Settings_English')
 prediction2 = status.predict(df2.review)
 print(prediction2)
 df2['VC-Settings'] = prediction2
 return df2

def Volume(df2):
 volume = joblib.load('SaveModels/Model_Volume_English')
 prediction3 = volume.predict(df2.review)
 print(prediction3)
 df2['Volume'] = prediction3
 return df2