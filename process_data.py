import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Шаг 1: Загрузка данных
def load_data(input_path):
    print("Loading data...")
    df = pd.read_csv(input_path)
    print("Columns in dataset:", df.columns)  # Проверка названий столбцов
    return df


# Шаг 2: Предварительная обработка данных
def preprocess_data(df):
    print("Preprocessing data...")
    # Удаление строк с пропущенными значениями
    df = df.dropna()
    return df


# Шаг 3: Масштабирование данных
def scale_data(df, target_column):
    print("Scaling data...")
    # Проверка корректности названия целевого столбца
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Шаг 4: Разделение данных на обучающую и тестовую выборки
def split_data(X, y, test_size=0.2, random_state=42):
    print("Splitting data...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Шаг 5: Сохранение обработанных данных
def save_data(X_train, X_test, y_train, y_test, output_path):
    print("Saving processed data...")
    pd.DataFrame(X_train, columns=['Income', 'Age', 'Loan', 'Loan to Income']).to_csv(f"{output_path}/X_train.csv",
                                                                                      index=False)
    pd.DataFrame(X_test, columns=['Income', 'Age', 'Loan', 'Loan to Income']).to_csv(f"{output_path}/X_test.csv",
                                                                                     index=False)
    pd.DataFrame(y_train, columns=['Default']).to_csv(f"{output_path}/y_train.csv", index=False)
    pd.DataFrame(y_test, columns=['Default']).to_csv(f"{output_path}/y_test.csv", index=False)


if __name__ == "__main__":
    input_path = "asset_v1_Skillfactory+URFUML2023+SEP2023+type@asset+block@Credit.csv"  # Исходный файл
    output_path = "processed_data"  # Папка для сохранения результата
    target_column = "Default"  # Целевая переменная

    # Выполнение этапов
    df = load_data(input_path)
    df = preprocess_data(df)
    X, y = scale_data(df, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Создание директории для сохранения результатов
    import os

    os.makedirs(output_path, exist_ok=True)

    save_data(X_train, X_test, y_train, y_test, output_path)
