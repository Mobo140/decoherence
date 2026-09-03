# Quantum Decoherence Predictor — ежедневные команды.
#   make            список
#   make app        UI  http://localhost:7860
#   make test
#   make e8         smoke (--fast). Полный E6/E8 перезапишет CSV чемпионов.

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTEST := $(PYTHON) -m pytest tests/ -q

.DEFAULT_GOAL := help
.PHONY: help install app test \
	paper1 paper2 paper1-full paper2-full paper2-improved \
	e1 e2 e3 e4 e5 e6 e7 e8 e9 e10a e10b e11 e12 e13 e14 \
	e6-full e8-full pdf pdf-1 pdf-2

help: ## список команд
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## pip install -r requirements.txt в .venv
	$(PYTHON) -m pip install -r requirements.txt

app: ## workbench http://localhost:7860
	$(PYTHON) app.py

test: ## pytest tests/
	$(PYTEST)

# --- статьи: --fast по умолчанию ------------------------------------------

paper1: ## Paper 1 (E1–E5) smoke
	$(PYTHON) experiments/run_paper1.py --fast

paper2: ## Paper 2 (E1/E3/E4 на D,E) smoke
	$(PYTHON) experiments/run_paper2.py --fast

paper1-full: ## Paper 1 полный прогон
	$(PYTHON) experiments/run_paper1.py

paper2-full: ## Paper 2 полный прогон
	$(PYTHON) experiments/run_paper2.py

paper2-improved: ## E6 suite smoke
	$(PYTHON) experiments/run_paper2_improved.py --fast

# --- eN: e1–e5 без --fast; e6–e14 smoke. CSV чемпионов: e6/e8-full --------

e1: ## ablation A–C
	$(PYTHON) -m experiments.e1_ablation

e2: ## window sweep
	$(PYTHON) -m experiments.e2_window_sweep

e3: ## noise sweep
	$(PYTHON) -m experiments.e3_noise_sweep

e4: ## cross-system
	$(PYTHON) -m experiments.e4_cross_system

e5: ## inverse baseline
	$(PYTHON) -m experiments.e5_inverse_baseline

e6: ## 2q improved smoke
	$(PYTHON) -m experiments.e6_2qubit_improved --fast

e7: ## Transformer / τ / J smoke
	$(PYTHON) -m experiments.e7_paper2_extended --fast

e8: ## improved 2q smoke — тоже пишет e8_*.csv
	$(PYTHON) -m experiments.e8_improved_2qubit --fast

e9: ## context codes smoke
	$(PYTHON) -m experiments.e9_context_injection --fast

e10a: ## fixed context smoke
	$(PYTHON) -m experiments.e10_fixed_context --fast

e10b: ## TFIM specialized smoke
	$(PYTHON) -m experiments.e10_tfim_specialized --fast

e11: ## scaling smoke
	$(PYTHON) -m experiments.e11_scaling --fast

e12: ## stretched-exp baseline smoke
	$(PYTHON) -m experiments.e12_baseline_comparison --fast

e13: ## survival TFIM smoke
	$(PYTHON) -m experiments.e13_survival_tfim --fast

e14: ## inject J smoke
	$(PYTHON) -m experiments.e14_inject_J --fast

e6-full: ## полный E6 — перезапишет e6_*.csv
	$(PYTHON) -m experiments.e6_2qubit_improved

e8-full: ## полный E8 — перезапишет e8_*.csv чемпионов
	$(PYTHON) -m experiments.e8_improved_2qubit

# --- PDF статей (latexmk). Не путать с paper1/paper2 = прогон экспериментов.

pdf: pdf-1 pdf-2 ## собрать paper_1 и paper_2 PDF

pdf-1: ## paper/paper_1/main.pdf
	cd paper/paper_1 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

pdf-2: ## paper/paper_2/paper2.pdf
	cd paper/paper_2 && latexmk -pdf -interaction=nonstopmode -halt-on-error paper2.tex
