# Quantum Decoherence Predictor — ежедневные команды.
#   make            список
#   make app        UI  http://localhost:7860
#   make test
#   make golden     эталонные выходы (должно быть "identical")
#   make e8         smoke (--fast) пишет *_fast.csv; полный E6/E8 перезапишет CSV чемпионов.

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTEST := $(PYTHON) -m pytest tests/ -q

.DEFAULT_GOAL := help
.PHONY: help install app test golden \
	paper1 paper1-full paper2-improved \
	e1 e2 e3 e4 e5 e6 e7 e8 e11 \
	e6-full e8-full pdf pdf-1 pdf-2

help: ## список команд
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## pip install -r requirements.txt в .venv
	$(PYTHON) -m pip install -r requirements.txt

app: ## workbench http://localhost:7860
	$(PYTHON) app.py

test: ## pytest tests/
	$(PYTEST)

golden: ## эталонные выходы scripts/golden_outputs.json
	$(PYTHON) -m scripts.golden_outputs

# --- статьи: --fast по умолчанию; smoke пишет *_fast.csv --------------------

paper1: ## Paper 1 (E1–E5) smoke
	$(PYTHON) -m experiments.run_paper1 --fast

paper1-full: ## Paper 1 полный прогон
	$(PYTHON) -m experiments.run_paper1

paper2-improved: ## E6 suite smoke
	$(PYTHON) -m experiments.run_paper2_improved --fast

# --- eN: e1–e5 полный прогон (--fast даёт smoke); e6–e11 smoke; чемпионы: e6/e8-full

e1: ## ablation A–C
	$(PYTHON) -m experiments.paper1.e1_ablation

e2: ## window sweep
	$(PYTHON) -m experiments.paper1.e2_window_sweep

e3: ## noise sweep
	$(PYTHON) -m experiments.paper1.e3_noise_sweep

e4: ## cross-system
	$(PYTHON) -m experiments.paper1.e4_cross_system

e5: ## inverse baseline
	$(PYTHON) -m experiments.paper1.e5_inverse_baseline

e6: ## 2q improved smoke
	$(PYTHON) -m experiments.paper2.e6_2qubit_improved --fast

e7: ## Transformer / τ / J smoke
	$(PYTHON) -m experiments.paper2.e7_paper2_extended --fast

e8: ## improved 2q smoke — тоже пишет e8_*.csv
	$(PYTHON) -m experiments.paper2.e8_improved_2qubit --fast




e11: ## scaling smoke
	$(PYTHON) -m experiments.paper2.e11_scaling --fast




e6-full: ## полный E6 — перезапишет e6_*.csv
	$(PYTHON) -m experiments.paper2.e6_2qubit_improved

e8-full: ## полный E8 — перезапишет e8_*.csv чемпионов
	$(PYTHON) -m experiments.paper2.e8_improved_2qubit

# --- PDF статей (latexmk). Не путать с paper1/paper2-improved = прогон экспериментов.

pdf: pdf-1 pdf-2 ## собрать paper_1 и paper_2 PDF

pdf-1: ## paper/paper_1/main.pdf
	cd paper/paper_1 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

pdf-2: ## paper/paper_2/main.pdf
	cd paper/paper_2 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

pdf-ru: ## paper/description_ru.pdf (описание обеих работ на русском)
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error description_ru.tex
