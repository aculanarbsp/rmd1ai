dataset_ = "2YR"
n_steps = 15
n_steps_ahead = 4
batch_size = 150
max_epochs = 100


from src.model_functions.model_builder import build_model
from keras._tf_keras.keras import callbacks
from src.functions import reformat_to_arrays, split_data, scale_the_data, read_data, cross_val, train_model

df = read_data(dataset_)

# The split_data function divides our dataframe yields into train, validation, and test sets

train, val, test = split_data(df, train_pct = 0.7, val_pct = 0.1)
print(f"The training size is {len(train)}")

train_scaled, train_mean, train_std, scaler_for_training_fitted_ = scale_the_data(train)
val_scaled, val_mean, val_std, scaler_for_val_fitted_= scale_the_data(val)
test_scaled, test_mean, test_std, scaler_for_test_fitted_ = scale_the_data(test)

X_train_scaled_, y_train_scaled_ = reformat_to_arrays(train_scaled, n_steps=n_steps, n_steps_ahead=n_steps_ahead)
X_val_scaled_, y_val_scaled_ = reformat_to_arrays(val_scaled, n_steps=n_steps, n_steps_ahead=n_steps_ahead)
X_test_scaled_, y_test_scaled_ = reformat_to_arrays(test_scaled, n_steps=n_steps, n_steps_ahead=n_steps_ahead)
print(f"The length of X_train is {len(X_train_scaled_)} and the length of y_train is {len(y_train_scaled_)}")

train_val_test_dict = {
    'train': 
        {'dataframe': train, 'scaler': scaler_for_training_fitted_, 
         'scaler_mean': train_mean, 'scaler_std': train_std, 
         'X_scaled': X_train_scaled_, 'y_scaled': y_train_scaled_},
    'val': 
        {'dataframe': val, 'scaler': scaler_for_val_fitted_, 
         'scaler_mean': val_mean, 'scaler_std': val_std,
         'X_scaled': X_val_scaled_, 'y_scaled': y_val_scaled_},
    'test': 
        {'dataframe': test, 'scaler': scaler_for_test_fitted_, 
         'scaler_mean': test_mean, 'scaler_std': test_std, 
         'X_scaled': X_test_scaled_, 'y_scaled': y_test_scaled_}
    }

rnn = build_model(
    model_ = 'rnn',
    neurons = 25,
    l1_reg = 0.01,
    seed = 0,
    n_steps = 15,
    n_steps_ahead = 4, x_train_=X_train_scaled_
    )

es = callbacks.EarlyStopping(
        monitor='loss', 
        mode='min', 
        verbose=1, 
        patience=100,
        min_delta=1e-7,
        restore_best_weights=True
    )

rnn.summary()
# print(f"size of training is {len(train)}")
# print(f"With a shape of: {X_train_scaled_.shape}")
print(X_train_scaled_.shape[1], X_train_scaled_.shape[-1])

rnn.fit(train_val_test_dict['train']['X_scaled'], train_val_test_dict['train']['y_scaled'], epochs=max_epochs, validation_data=(train_val_test_dict['val']['y_scaled'], train_val_test_dict['val']['y_scaled']),
                  batch_size=batch_size, callbacks=[es], shuffle=False)