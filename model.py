import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def get_data():
    train = pd.read_csv('internship_train.csv')
    test = pd.read_csv('internship_hidden_test.csv')
    return train, test

def get_combined_data():
    # reading train data
    train, test = get_data()

    target = train.target
    train.drop(['target'], axis=1, inplace=True)

    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index'], inplace=True, axis=1)
    return combined, target


def get_cols_with_no_nans(df, col_type):
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else:
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


def split_combined():
    global combined
    train = combined[:90000]
    test = combined[90000:]
    return train, test


def make_submission(prediction, sub_name):
    nums = [num for num in range(0, 53)]
    cols = [f'{str(n)}' for n in nums]
    my_submission = pd.DataFrame(columns=cols)
    for i in nums:
        my_submission[f'{str(i)}'] = test[f'{str(i)}']
    my_submission['target'] = prediction
    my_submission.to_csv('{}.csv'.format(sub_name), index=False)
    print('A submission file has been made')


def drop_feature(data):
    data_without_8 = data.drop(['8'], axis=1)
    scaler = StandardScaler()
    return scaler.fit_transform(data_without_8)


if __name__ == '__main__':
    train_data, test_data = get_data()
    combined, target = get_combined_data()

    num_cols = get_cols_with_no_nans(combined, 'num')
    cat_cols = get_cols_with_no_nans(combined, 'no_num')

    combined = combined[num_cols + cat_cols]

    train_data = train_data[num_cols + cat_cols]
    train_data['target'] = target

    train, test = split_combined()

    # Build NN Model
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal', input_dim=train.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=RootMeanSquaredError())
    NN_model.summary()

    # checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    filepath = 'weights_best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    NN_model.fit(train, target, epochs=50, batch_size=32, validation_split=0.2, callbacks=callbacks_list)

    # Load wights file of the best model :
    # wights_file = 'Weights-050--0.05109.hdf5'     # choose the best checkpoint
    # NN_model.load_weights(wights_file)    # load it
    NN_model.load_weights(filepath)
    NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=RootMeanSquaredError())

    predictions = NN_model.predict(test)
    make_submission(predictions[:, 0], 'submission_NN')

    # Drop dependencies and normalize data
    normalized_train = drop_feature(train)
    normalized_test = drop_feature(test)
    # Build Random Rorest Model
    X_train, X_val, y_train, y_val = train_test_split(normalized_train, target, test_size=0.2, random_state=1)

    model = RandomForestRegressor(verbose=2, n_estimators=20, random_state=1)
    model.fit(X_train, y_train)

    # Get the RMSE on the validation data
    predicted_prices = model.predict(X_val)
    RMSE = np.sqrt(mean_squared_error(y_val, predicted_prices))
    print('Random forest validation RMSE = ', RMSE)

    predicted_prices = model.predict(normalized_test)
    make_submission(predicted_prices, 'Submission_RF')
