# Telegram Style Transfer Test Task

Решение тестового задания по адаптации LLM под два разных Telegram-стиля:

- **Type 1** — стиль канала [banki_oil](https://t.me/banki_oil)
- **Type 2** — стиль канала [moscowach](https://t.me/moscowach)

## Что внутри

- `tel_style.ipynb` — основной запуск пайплайна
- `src/telegram_style/` — модульная логика
- `RESULTS.md` — краткое описание шагов, ограничений и результатов
- `outputs_type1.txt` — финальные ответы для `inputs_type1.txt`
- `outputs_type2.txt` — финальные ответы для `inputs_type2.txt`

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

- данные разделены на **train / test** 
- baseline считается до подстройки
- затем обучаются две отдельные **LoRA**
- качество сравнивается на held-out test примерах
- проверяется не только “до / после”, но и то, что каждый стиль остаётся ближе именно к своему целевому типу
