import pandas as pd

# Загрузка данных (лучше делать это один раз при старте приложения) :)
try:
    df = pd.read_csv('archive/freelancer_earnings_bd.csv')
except FileNotFoundError:
    print(
        "Ошибка: Файл с данными не найден. Убедитесь, что 'freelancer_earnings_bd.csv' находится в нужной директории.")
    df = pd.DataFrame()  # Создаем пустой DataFrame, чтобы избежать ошибок далее


def handle_error(message="Неизвестная ошибка при обработке запроса."):
    return message


def compare_average(data_frame, measure_column, category_column, target_category_value):
    if data_frame.empty:
        return "Данные не загружены."
    if not all(col in data_frame.columns for col in [measure_column, category_column]):
        return f"Ошибка: Один или несколько столбцов ({measure_column}, {category_column}) не найдены в данных."
    if target_category_value not in data_frame[category_column].unique():
        return f"Ошибка: Значение '{target_category_value}' не найдено в столбце '{category_column}'."

    try:
        avg_target_earnings = data_frame[data_frame[category_column] == target_category_value][measure_column].mean()
        avg_other_earnings = data_frame[data_frame[category_column] != target_category_value][measure_column].mean()

        if pd.isna(avg_target_earnings) or pd.isna(avg_other_earnings):
            return "Недостаточно данных для сравнения одной из групп."

        result_str = f"Среднее значение '{measure_column}' для '{category_column}' = '{target_category_value}': {avg_target_earnings:,.2f}\n"
        result_str += f"Среднее значение '{measure_column}' для других категорий в '{category_column}': {avg_other_earnings:,.2f}\n"

        if avg_other_earnings > 0:
            difference = avg_target_earnings - avg_other_earnings
            percentage_difference = (difference / avg_other_earnings) * 100
            if difference > 0:
                result_str += f"Значение для '{target_category_value}' в среднем на {difference:,.2f} ({percentage_difference:.2f}%) выше."
            elif difference < 0:
                result_str += f"Значение для '{target_category_value}' в среднем на {abs(difference):,.2f} ({abs(percentage_difference):.2f}%) ниже."
            else:
                result_str += f"Значения в среднем одинаковы."
        else:
            result_str += "Невозможно рассчитать процентное соотношение, т.к. среднее значение для других категорий равно нулю или не определено."
        return result_str
    except Exception as e:
        return f"Ошибка при выполнении compare_average: {e}"


def group_by_aggregate(data_frame, group_by_column, aggregate_column, aggregations):
    if data_frame.empty:
        return "Данные не загружены."
    if not all(col in data_frame.columns for col in [group_by_column, aggregate_column]):
        return f"Ошибка: Один или несколько столбцов ({group_by_column}, {aggregate_column}) не найдены в данных."

    valid_aggregations = [agg for agg in aggregations if agg in ['mean', 'median', 'sum', 'count', 'min', 'max', 'std']]
    if not valid_aggregations:
        return "Ошибка: Не указаны допустимые функции агрегации (например, 'mean', 'median')."

    try:
        grouped_data = data_frame.groupby(group_by_column)[aggregate_column].agg(valid_aggregations)
        # Для лучшего форматирования вывода, особенно если это идет в CLI
        return f"Агрегированные данные для '{aggregate_column}' по группам из '{group_by_column}':\n{grouped_data.to_string()}"
    except Exception as e:
        return f"Ошибка при выполнении group_by_aggregate: {e}"


def filter_and_calculate_percentage(data_frame, base_filter_column, base_filter_value,
                                    condition_column, condition_operator, condition_value, value_is_numeric):
    if data_frame.empty:
        return "Данные не загружены."
    if not all(col in data_frame.columns for col in [base_filter_column, condition_column]):
        return f"Ошибка: Один или несколько столбцов ({base_filter_column}, {condition_column}) не найдены в данных."

    try:
        # Преобразование condition_value к нужному типу
        if value_is_numeric:
            try:
                # Попытка преобразовать в float, если это число с плавающей точкой, или int
                if '.' in str(condition_value) or 'e' in str(condition_value).lower():
                    condition_value = float(condition_value)
                else:
                    condition_value = int(condition_value)
            except ValueError:
                return f"Ошибка: не удалось преобразовать '{condition_value}' в число."

        # Фильтр для общей группы (знаменатель)
        base_group_df = data_frame[data_frame[base_filter_column] == base_filter_value]
        total_in_base_group = len(base_group_df)

        if total_in_base_group == 0:
            return f"Нет данных для базового фильтра: '{base_filter_column}' = '{base_filter_value}'."

        # Применение условия для числителя
        if condition_operator == '<':
            filtered_df = base_group_df[base_group_df[condition_column] < condition_value]
        elif condition_operator == '>':
            filtered_df = base_group_df[base_group_df[condition_column] > condition_value]
        elif condition_operator == '==':
            filtered_df = base_group_df[base_group_df[condition_column] == condition_value]
        elif condition_operator == '!=':
            filtered_df = base_group_df[base_group_df[condition_column] != condition_value]
        elif condition_operator == '<=':
            filtered_df = base_group_df[base_group_df[condition_column] <= condition_value]
        elif condition_operator == '>=':
            filtered_df = base_group_df[base_group_df[condition_column] >= condition_value]
        else:
            return f"Ошибка: Неизвестный оператор '{condition_operator}'."

        count_in_condition = len(filtered_df)
        percentage = (count_in_condition / total_in_base_group) * 100 if total_in_base_group > 0 else 0

        return (f"Из {total_in_base_group} записей, где '{base_filter_column}' = '{base_filter_value}', "
                f"{count_in_condition} удовлетворяют условию '{condition_column} {condition_operator} {condition_value}'. "
                f"Это составляет {percentage:.2f}%.")
    except Exception as e:
        return f"Ошибка при выполнении filter_and_calculate_percentage: {e}"


def get_descriptive_stats(data_frame, column_name, group_by_column=None):
    if data_frame.empty:
        return "Данные не загружены."
    if column_name not in data_frame.columns:
        return f"Ошибка: Столбец '{column_name}' не найден."
    if data_frame[column_name].dtype not in ['int64', 'float64']:
        return f"Ошибка: Столбец '{column_name}' не является числовым."

    try:
        if group_by_column:
            if group_by_column not in data_frame.columns:
                return f"Ошибка: Столбец для группировки '{group_by_column}' не найден."
            stats = data_frame.groupby(group_by_column)[column_name].describe()
            return f"Описательные статистики для '{column_name}' сгруппированные по '{group_by_column}':\n{stats.to_string()}"
        else:
            stats = data_frame[column_name].describe()
            return f"Описательные статистики для '{column_name}':\n{stats.to_string()}"
    except Exception as e:
        return f"Ошибка при выполнении get_descriptive_stats: {e}"


# Основной диспетчер, который будет вызываться после получения JSON от LLM
def execute_analysis(json_query):
    try:
        operation = json_query.get("operation_type")
        params = json_query.get("parameters", {})

        if operation == "compare_average":
            return compare_average(df, params.get("measure_column"), params.get("category_column"),
                                   params.get("target_category_value"))
        elif operation == "group_by_aggregate":
            return group_by_aggregate(df, params.get("group_by_column"), params.get("aggregate_column"),
                                      params.get("aggregations", ['mean', 'median', 'count', 'sum']))
        elif operation == "filter_and_calculate_percentage":
            return filter_and_calculate_percentage(df, params.get("base_filter_column"),
                                                   params.get("base_filter_value"),
                                                   params.get("condition_column"), params.get("condition_operator"),
                                                   params.get("condition_value"), params.get("value_is_numeric", False))
        elif operation == "get_descriptive_stats":
            return get_descriptive_stats(df, params.get("column_name"), params.get("group_by_column"))
        elif operation == "error":
            return f"LLM не смогла обработать запрос: {params.get('message', 'Нет деталей')}"
        else:
            return f"Неизвестный тип операции: {operation}"
    except Exception as e:
        return f"Критическая ошибка при разборе JSON или вызове функции: {e}"