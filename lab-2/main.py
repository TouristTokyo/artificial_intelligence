import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

month = 8
key = "spark"
file_name = "wine_Austral.dat"
colors = ["b-", "g-"]
labels = ["Исходные данные", "Прогнозирование"]


def show_time_series(data):
    result = seasonal_decompose(data, model='additive', period=12)
    result.plot()
    plt.show()


def predict_the_result(train):
    model = SARIMAX(train, order=(3, 0, 0), seasonal_order=(0, 1, 0, 12))
    result = model.fit()

    start_point = len(train)
    end_point = start_point + month

    return result.predict(start_point, end_point)


if __name__ == '__main__':
    spark_data = pd.read_csv(file_name, sep="\t")[key]
    print(spark_data)
    show_time_series(spark_data)
    predicted_data = predict_the_result(spark_data)
    result = [spark_data, predicted_data]

    for i in range(len(result)):
        plt.plot(result[i], colors[i], label=labels[i])
    plt.ylabel('Кол-во игристого вина (тыс. литров)')
    plt.xlabel('Месяца')
    plt.legend()
    plt.grid()
    plt.show()
