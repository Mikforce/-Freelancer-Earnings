import pandas as pd

# Указываем путь к файлу
file_path = 'archive/freelancer_earnings_bd.csv'

try:
    # Загружаем CSV файл в DataFrame
    df = pd.read_csv(file_path)

    # Посмотрим на первые несколько строк
    print("Первые 5 строк данных:")
    print(df.head())
    print("\n" + "="*50 + "\n")

    # Получим общую информацию о DataFrame (типы данных, пропуски)
    print("Информация о DataFrame:")
    df.info()
    print("\n" + "="*50 + "\n")

    # Посмотрим на основные статистические показатели для числовых столбцов
    print("Основные статистические показатели:")
    print(df.describe())
    print("\n" + "="*50 + "\n")

    # Посмотрим на названия столбцов
    print("Названия столбцов:")
    print(df.columns.tolist())
    print("\n" + "="*50 + "\n")

    # Посмотрим на уникальные значения в некоторых категориальных столбцах,
    # чтобы лучше понять их содержимое
    categorical_cols_to_inspect = [
        'Currency', 'Expertise_Level', 'Job_Type',
        'Country', 'Preferred_Payment_Method', 'Satisfaction_Score'
    ]
    for col in categorical_cols_to_inspect:
        if col in df.columns:
            print(f"Уникальные значения в столбце '{col}':")
            print(df[col].value_counts(dropna=False)) # dropna=False покажет и количество NaN, если есть
            print("-" * 30)

except FileNotFoundError:
    print(f"Ошибка: Файл '{file_path}' не найден. Пожалуйста, проверьте путь к файлу.")
except Exception as e:
    print(f"Произошла ошибка при загрузке или обработке файла: {e}")