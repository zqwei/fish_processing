# Spike detection for voltron data

# Example

```
window_length = 40
x_, contain_outliers_ = prepare_sequences(voltr, spkcount, window_length)
train_test_index = np.random.rand(x_.shape[0])>0.3
x_test = x_[~train_test_index, :, :]
y_test = contain_outliers_[~train_test_index][:, np.newaxis]
x_train = x_[train_test_index, :, :]
y_train = contain_outliers_[train_test_index][:, np.newaxis]
hidden_dim = 100
m = create_lstm_model(hidden_dim, window_length)
m.fit(x_train, y_train, batch_size=128, nb_epoch=5, validation_data=(x_test, y_test))
pred_x_test = m.predict(x_test)
```
