import pandas as pd
from RF_Model import Model,Predict

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)
    df = pd.read_excel('test.xlsx')
    ### Training and saving the Model
    Model(df)
    ### Auto prediction using saved model
    df2 = pd.read_excel('Prediction data.xlsx')
    print(df2)
    df3= Predict(df2)
    print(df3)
    df3.to_excel('Final_Auto_Prediction.xlsx')