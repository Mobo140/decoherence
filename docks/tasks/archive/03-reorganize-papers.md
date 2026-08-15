# Реорганизация статей: paper/paper_1/ и paper/paper_2/

## Task

Переместить файлы статей в отдельные папки:
- `paper/main.tex` → `paper/paper_1/main.tex`
- `paper/paper2.tex` → `paper/paper_2/paper2.tex`
Обновить все ссылки и crossrefs между статьями.

## Why

Сейчас обе статьи лежат в одной папке `paper/`, что создаёт путаницу.
При добавлении figures, supplementary, bibliography — файлы будут перемешиваться.
Чёткое разделение: paper_1 (1-qubit) и paper_2 (2-qubit) — стандарт для публикации.

## Scope

**Входит:**
- Создать `paper/paper_1/` и `paper/paper_2/`
- Переместить `.tex` файлы (main.tex → paper_1/, paper2.tex → paper_2/)
- Создать `paper/paper_1/figures/` и `paper/paper_2/figures/` (пустые, для будущих графиков)
- Обновить ссылки в paper2.tex на paper1 (`\cite{Paper1Brusnikin}` остаётся)
- Обновить `docks/main/ARCHITECTURE.md` — пути к статьям
- Обновить `docks/main/PROJECT_PLAN.md` — пути к статьям
- Обновить `docks/current/CURRENT_STATUS.md`

**Не входит:**
- Изменение содержимого статей
- Добавление/обновление метрик
- Компиляция PDF

## Acceptance Criteria

- [x] `paper/paper_1/main.tex` существует
- [x] `paper/paper_2/paper2.tex` существует
- [x] `paper/paper_1/figures/` создана
- [x] `paper/paper_2/figures/` создана
- [x] Старые файлы `paper/main.tex` и `paper/paper2.tex` удалены (или заменены symlink)
- [x] Все документы в `docks/` ссылаются на новые пути

## Verification

```bash
ls paper/paper_1/
ls paper/paper_2/
# ожидаем: main.tex, figures/
# ожидаем: paper2.tex, figures/
```

## Status

`done`

Корневые `paper/main.tex` и `paper/paper2.tex` — symlink на `paper_1/` и `paper_2/`.
Текст статей не менялся. `paper_2/paper2.tex` новее корневой копии (E6 figures + E11a).

## Priority

P1
