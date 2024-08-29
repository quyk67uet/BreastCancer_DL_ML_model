import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import tensorflow as tf
from tensorflow import keras

def create_model(data): 
    X = data.drop(columns='diagnosis', axis=1)
    Y = data['diagnosis']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    tf.random.set_seed(3)

    model = keras.Sequential([
        keras.layers.Input(shape=(30,)),  # Sử dụng keras.layers.Input thay vì Flatten
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Sử dụng sigmoid cho bài toán phân loại nhị phân
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Sử dụng binary_crossentropy cho bài toán phân loại nhị phân
                  metrics=['accuracy'])

    model.fit(X_train_std, Y_train, epochs=10, batch_size=32)  # Thêm epochs và batch_size

    y_pred_prob = model.predict(X_test_std)
    y_pred = (y_pred_prob > 0.5).astype("int32")  # Chuyển đổi xác suất thành nhãn lớp

    print('Accuracy of our model: ', accuracy_score(Y_test, y_pred))
    print("Classification report: \n", classification_report(Y_test, y_pred))

    return model, scaler

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 0, 'B': 1})
    return data

def main():
    data = get_clean_data()
    model, scaler = create_model(data)

    # Lưu mô hình Keras
    model.save('DL/model_DL.h5')
    
    # Lưu scaler
    with open('DL/scaler_DL.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()