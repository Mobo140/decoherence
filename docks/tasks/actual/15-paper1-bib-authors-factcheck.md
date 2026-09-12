# Task: Paper 1 — библиография, авторы, факт-чек

## Why

Перед сабмитом `paper/paper_1/main.tex` есть три блокера, не связанных с рисунками:
битая и тонкая библиография, отсутствие блока авторов, и одна фактическая
нестыковка abstract↔таблица.

## Scope

**Входит:** только `paper/paper_1/main.tex`.

- **Библиография**: `\bibitem{TransformerLindblad2025}` начинается с запятой и **без
  авторов** — починить; расширить с 4 до ~10–12 ссылок (Lindblad/open systems, T₂/
  decoherence, QEC, PINN, BiLSTM, ML-for-open-quantum, Bernstein-параметризация,
  AUROC/калибровка). Ключ согласовать с Paper 2 (там цитируется `Paper1Brusnikin`).
- **Авторы**: добавить `\author`/`\affil` (`authblk` уже подключён, `\author` нет) —
  Nikita Brusnikin + аффилиация.
- **Факт-чек**: abstract утверждает «inverse хуже в >15× по MAE», а таблица E5 даёт
  3–9× (max 9.3× при f=0.5) → исправить число. Явно проговорить claims C3/C4/C5.
  Смягчить MAE-историю (MAE на B ≈0.356, держимся на R²/AUROC).

**Не входит:**
- Рисунки (задача 14), перезапуск экспериментов, revtex-миграция.

## Acceptance Criteria

- [ ] `\bibitem{TransformerLindblad2025}` с корректными авторами; ≥10 ссылок, все
  цитируются в тексте (нет висячих `\cite`/`\bibitem`)
- [ ] Блок авторов и аффилиация присутствуют; `\maketitle` рендерит автора
- [ ] Абстрактный множитель inverse-vs-direct совпадает с таблицей E5
- [ ] Claims C1–C5 явно помечены в тексте
- [ ] `make pdf-1` компилируется без warnings по неопределённым `\cite`

## Invariants

- IV1: Числа в таблицах E1–E5 не меняются (только текст/ссылки/факт-чек)

## Verification

```bash
make pdf-1
grep -c 'undefined' paper/paper_1/main.log   # -> 0 (нет undefined citations)
```

## Дополнительно (нужно от пользователя)

- Аффилиация автора (по умолчанию — поставлю placeholder до уточнения).

## Status

`planned`

## Priority

P1
