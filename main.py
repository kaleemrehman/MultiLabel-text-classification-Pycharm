import pandas as pd
import joblib

from Training_Functions import Connection_Training, Settings_Training ,Volume_Training
from Prediction_Functions import Connection, Vcsettings, Volume


if __name__ == '__main__':
    print("Please Wait Program is running will take some time")
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)
    ################################## TRAINING CODE#########################################
    ############# Load data with having Connection issues
    # df = pd.read_excel('Datasets/Negative_labeled_final.xlsx')
    # df = df[['review', 'Bluetooth', 'VC-Settings', 'Volume']]
    #
    #
    # ############# Save Connection model using Joblib
    # pipeline=Connection_Training(df)
    # joblib.dump(pipeline, 'SaveModels/Model_Conn_English')
    ############# Load data with having VcSettings (status) issues
    # df2 = pd.read_csv('Datasets/Statuslabel.csv')
    #
    # # ############# Save VcSettings (staus) model using Joblib
    # pipeline=Settings_Training(df2)
    # joblib.dump(pipeline, 'SaveModels/Model_Settings_English')
    # #
    # # ############# Load data with having Volume issues
    # df3 = pd.read_csv('Datasets/Volumelabel.csv')
    # ############# Save Volume model using Joblib
    # pipeline = Volume_Training(df3)
    # joblib.dump(pipeline, 'SaveModels/Model_Volume_English')

    ################################PREDICTION CODE####################################################
    df4 = pd.read_excel('Prediction Dataset/Prediction data.xlsx')
    df4 = df4[df4.stars < 3]
    df4= Connection(df4)
    df4 = Vcsettings(df4)
    df4 = Volume(df4)
    print(df4)
    ########## Export to Excel File

    df4.to_excel('Prediction Dataset/PredictionFinal.xlsx', index=False)