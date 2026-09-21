# Статьи

Источники только в подпапках. Корень `paper/` больше не собирается.

- `paper_1/main.tex` — Paper 1, сценарии A–C (1 кубит). Таблицы из `experiments/results/e1_*_p1.csv` … `e5_*.csv`.
- `paper_2/main.tex` — Paper 2, сценарии D/E (XXZ / TFIM). Рисунки в `paper_2/figures/`. Чемпионы: XXZ R² 0.817 (E11a), TFIM R² 0.575 (E8c).
- `description_ru.tex` — черновик на русском, не публикация. Таблица D/E там старая (до E8).

Сборка:

```bash
make pdf      # обе статьи
make pdf-1    # paper/paper_1/main.pdf
make pdf-2    # paper/paper_2/paper2.pdf
```

E9–E14 в `.tex` нет: они не побили E8c / E11a.
