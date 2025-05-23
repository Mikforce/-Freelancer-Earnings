import os
import json
import requests
from dotenv import load_dotenv
from llm_analyzer import execute_analysis, df

load_dotenv()

# --- Конфигурация DeepSeek ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"  # Базовый URL API

if not DEEPSEEK_API_KEY:
    print("Ошибка: API ключ DeepSeek не найден.")
    print("Пожалуйста, убедитесь, что у вас есть файл .env с DEEPSEEK_API_KEY=ваш_ключ")
    print("Или переменная окружения DEEPSEEK_API_KEY установлена.")
    exit()

# --- Промпт для LLM ---
SYSTEM_PROMPT = """
Ты — ИИ-ассистент, специализирующийся на анализе данных о фрилансерах.
Твоя задача — преобразовать вопрос пользователя на естественном языке в структурированный JSON-запрос.
Этот JSON-запрос будет использоваться для выполнения анализа данных с помощью Python-скрипта.

**Схема данных (DataFrame `df`):**
*   `Freelancer_ID`: int64, ID фрилансера.
*   `Job_Category`: object (категориальный), Категория работы (например, 'Web Development', 'App Development', 'Data Entry', 'Digital Marketing', 'Writing', 'Graphic Design', 'Customer Service', 'Video Editing', 'SEO', 'Virtual Assistant').
*   `Platform`: object (категориальный), Платформа, на которой работает фрилансер (например, 'Upwork', 'Freelancer', 'Fiverr', 'Guru', 'Toptal', 'PeoplePerHour').
*   `Experience_Level`: object (категориальный), Уровень опыта фрилансера. **Возможные значения: 'Beginner', 'Intermediate', 'Expert'**.
*   `Client_Region`: object (категориальный), Регион клиента. **Возможные значения: 'Australia', 'USA', 'Middle East', 'Asia', 'UK', 'Europe', 'Canada'**.
*   `Payment_Method`: object (категориальный), Способ оплаты. **Возможные значения: 'Crypto', 'Bank Transfer', 'PayPal', 'Mobile Banking'**.
*   `Job_Completed`: int64, Количество выполненных проектов.
*   `Earnings_USD`: int64, Общий доход в долларах США.
*   `Hourly_Rate`: float64, Почасовая ставка в USD.
*   `Job_Success_Rate`: float64, Процент успешно выполненных проектов (0-100).
*   `Client_Rating`: float64, Средний рейтинг от клиентов (обычно 1-5).
*   `Job_Duration_Days`: int64, Средняя продолжительность проекта в днях.
*   `Project_Type`: object (категориальный), Тип проекта (например, 'Fixed-Price', 'Hourly', 'Milestone-based', 'Recurring').
*   `Rehire_Rate`: float64, Процент повторных наймов (0-100).
*   `Marketing_Spend`: int64, Расходы на маркетинг.

**Форматы JSON-запросов для различных операций:**
1.  **`compare_average`**: Сравнить среднее значение числового столбца для двух групп, определенных по категориальному столбцу. Одна группа - целевое значение, вторая - все остальные.
    *   `operation_type`: "compare_average"
    *   `parameters`:
        *   `measure_column`: str (столбец для измерения среднего, например, "Earnings_USD")
        *   `category_column`: str (столбец для определения групп, например, "Payment_Method")
        *   `target_category_value`: str (значение в category_column для первой группы, например, "Crypto")

2.  **`group_by_aggregate`**: Сгруппировать данные по категориальному столбцу и рассчитать агрегаты (среднее, медиана, сумма, количество) для числового столбца.
    *   `operation_type`: "group_by_aggregate"
    *   `parameters`:
        *   `group_by_column`: str (столбец для группировки, например, "Client_Region")
        *   `aggregate_column`: str (числовой столбец для агрегации, например, "Earnings_USD")
        *   `aggregations`: list[str] (список агрегаций, например, ["mean", "median", "count", "sum"]. Допустимые: "mean", "median", "sum", "count", "min", "max", "std").

3.  **`filter_and_calculate_percentage`**: Отфильтровать данные по одному или нескольким условиям и рассчитать процент.
    *   `operation_type`: "filter_and_calculate_percentage"
    *   `parameters`:
        *   `base_filter_column`: str (столбец для основного фильтра/знаменателя, например, "Experience_Level")
        *   `base_filter_value`: any (значение для основного фильтра, например, "Expert")
        *   `condition_column`: str (столбец для дополнительного условия/числителя, например, "Job_Completed")
        *   `condition_operator`: str (оператор сравнения: "<", ">", "==", "!=", "<=", ">=", например, "<")
        *   `condition_value`: any (значение для дополнительного условия, например, 100)
        *   `value_is_numeric`: bool (true, если condition_value является числом, false если строка)

4.  **`get_descriptive_stats`**: Получить описательные статистики для числового столбца.
    *   `operation_type`: "get_descriptive_stats"
    *   `parameters`:
        *   `column_name`: str (название числового столбца, например, "Earnings_USD")
        *   `group_by_column`: str (опционально, столбец для группировки перед расчетом статистик, например, "Job_Category")

**Важно:**
*   Внимательно используй точные названия столбцов из схемы данных.
*   Для категориальных столбцов используй точные значения, указанные в схеме (например, 'Expert', не 'expert').
*   Если вопрос не может быть однозначно преобразован в одну из указанных операций или требует данных, которых нет в схеме, верни JSON: `{"operation_type": "error", "message": "Не могу обработать запрос"}`.
*   Числовые значения в параметрах `condition_value` должны быть числами, а не строками, если `value_is_numeric` это `true`.

Теперь преобразуй следующий вопрос пользователя в JSON:
"""


def get_llm_response(user_question):
    system_message_content = SYSTEM_PROMPT.split("Теперь преобразуй следующий вопрос пользователя в JSON:")[0].strip()
    user_message_content = f"Преобразуй следующий вопрос пользователя в JSON:\n{user_question}"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",  # Или "deepseek-coder", если он лучше для этой задачи
        "messages": [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ],
        "temperature": 0.1,
        "max_tokens": 500,
        # "response_format": {"type": "json_object"} # хз поддерживается ли это через прямой API вызов )))

        # "stream": False # Обычно для чат-комплишенов по умолчанию False
    }


    api_url = f"{DEEPSEEK_API_BASE_URL}/chat/completions"

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Проверка на HTTP ошибки (4xx или 5xx)

        response_data = response.json()

        # Стандартная структура ответа OpenAI-совместимых API
        json_string = response_data["choices"][0]["message"]["content"]

        # Очистка JSON строки, если LLM добавляет ```json ... ```
        if json_string.startswith("```json"):
            json_string = json_string[len("```json"):]
        if json_string.endswith("```"):
            json_string = json_string[:-len("```")]

        json_string = json_string.strip()

        parsed_json = json.loads(json_string)
        return parsed_json

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP ошибка: {http_err}")
        print(f"Ответ сервера: {response.text}")
        return {"operation_type": "error", "message": f"Ошибка API LLM (HTTP): {http_err}"}
    except requests.exceptions.RequestException as req_err:
        print(f"Ошибка запроса к API: {req_err}")
        return {"operation_type": "error", "message": f"Ошибка сети или подключения к API LLM: {req_err}"}
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON от LLM: {e}")
        print(f"Полученная строка: >>>{json_string}<<<")
        return {"operation_type": "error", "message": "Ошибка формата JSON от LLM."}
    except (KeyError, IndexError) as e:
        print(f"Неожиданная структура ответа от LLM: {e}")
        print(f"Ответ сервера: {response_data}")
        return {"operation_type": "error", "message": f"Неожиданный формат ответа от LLM: {e}"}
    except Exception as e:  # Общий обработчик на всякий случай
        print(f"Неизвестная ошибка при взаимодействии с LLM: {e}")
        return {"operation_type": "error", "message": f"Неизвестная ошибка API LLM: {e}"}


def main():
    if df.empty:
        print("Данные не были загружены. Программа не может продолжить работу.")
        return

    print("Прототип системы анализа данных о фрилансерах.")
    print("Введите ваш вопрос или 'выход' для завершения.")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'выход':
            break

        if not user_input.strip():
            continue

        print("\nОбработка запроса с помощью LLM...")
        llm_json_query = get_llm_response(user_input)

        print(f"LLM сгенерировала JSON:\n{json.dumps(llm_json_query, indent=2, ensure_ascii=False)}")

        if llm_json_query:  # llm_json_query всегда будет dict
            result = execute_analysis(llm_json_query)
            print("\nРезультат анализа:")
            print(result)
            print("-" * 50)



if __name__ == "__main__":
    main()