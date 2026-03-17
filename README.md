# Telegram Style Transfer Test Task

Решение тестового задания по адаптации LLM под два разных Telegram-стиля:

- **Type 1** — стиль канала [banki_oil](https://t.me/banki_oil)
- **Type 2** — стиль канала [moscowach](https://t.me/moscowach)

## Идея решения

Готовый датасет в задаче не выдавался, поэтому я сначала сам собрал корпуса каналов, а затем на их основе построил компактный датасет для style transfer:

1. спарсил сырые посты каналов;
2. случайно отобрал по 30 реальных постов на канал;
3. для этих постов получил нейтральные входы;
4. построил пары `нейтральный текст -> реальный пост канала`;
5. разделил их на **20 train / 10 test**;
6. обучил две отдельные LoRA;
7. сгенерировал финальные ответы и сравнил baseline vs adapted на held-out test.

## Что внутри

### Основной запуск
- `tel_style.ipynb` — основной notebook с полным пайплайном
- `src/telegram_style/` — модульная логика

### Отчёт и результаты
- `RESULTS.md` — описание шагов, ограничений, метрик и итогов
- `outputs_type1.txt` — финальные ответы для `inputs_type1.txt`
- `outputs_type2.txt` — финальные ответы для `inputs_type2.txt`

### Данные, собранные для решения
- `banki_oil.txt` — сырой корпус постов канала
- `moscowach.txt` — сырой корпус постов канала
- `outputs_banki_oil.txt` — выбранные реальные посты `banki_oil` для мини-датасета
- `outputs_moscowach.txt` — выбранные реальные посты `moscowach` для мини-датасета
- `inputs_banki_oil.txt` — нейтральные версии выбранных постов `banki_oil`
- `inputs_moscowach.txt` — нейтральные версии выбранных постов `moscowach`

## Быстрые ссылки

- [RESULTS.md](RESULTS.md)
- [Ноутбук / основной запуск](tel_style.ipynb)
- [outputs_type1.txt](outputs_type1.txt)
- [outputs_type2.txt](outputs_type2.txt)

## Графики

- [Cosine similarity](plot_cosine_similarity.png)
- [Cross-style heatmap](plot_cross_style_heatmap.png)
- [Style gap](plot_style_margin.png)
- [Marker compliance](plot_style_compliance.png)

## Коротко о подходе

- датасет был **собран самостоятельно из Telegram-каналов**
- для ускорения использована уменьшенная выборка: 30 примеров на канал
- обучение и проверка разделены на **train / test**
- baseline считается **до** подстройки
- затем обучаются две отдельные **LoRA**
- качество сравнивается на held-out test примерах
- дополнительно проверяется, что каждый стиль остаётся ближе именно к своему целевому типу, а не к “среднему” стилю
