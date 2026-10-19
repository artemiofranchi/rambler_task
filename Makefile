PY ?= python3
CONFIG ?= configs/config.yaml
export MODERATION_CONFIG := $(CONFIG)
MODELS_URL ?=

.PHONY: help install data train-linear train-transformer calibrate train evaluate benchmark serve test \
        loadtest report all smoke smoke-run download-models package-models docker-build docker-run

help:            ## список команд
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install:         ## зависимости (обучение, тесты, нагрузка)
	$(PY) -m pip install -r requirements-dev.txt

data:            ## скачать датасет с Hugging Face и подготовить train/val/test
	$(PY) scripts/prepare_data.py

train-linear:    ## TF-IDF + LogReg (~2–5 мин на CPU)
	$(PY) scripts/train_linear.py

train-transformer: ## дообучить rubert-tiny2 (~15–30 мин на Apple Silicon / GPU)
	$(PY) scripts/train_transformer.py

calibrate:       ## веса ансамбля и пороги ALLOW/REVIEW/BLOCK на валидации
	$(PY) scripts/calibrate.py

train: train-linear train-transformer calibrate  ## обучить всё

evaluate:        ## метрики качества и устойчивости на тесте → reports/metrics.md
	$(PY) scripts/evaluate.py

benchmark:       ## офлайн-скорость → reports/benchmark.json
	$(PY) scripts/benchmark.py

serve:           ## запустить API на :8000
	$(PY) -m uvicorn moderation.api:app --host 0.0.0.0 --port 8000

test:            ## unit + API + интеграционные тесты
	$(PY) -m pytest

loadtest:        ## поднять сервис, нагрузить, остановить → reports/loadtest_*.json
	PY=$(PY) bash scripts/run_loadtest.sh

report:          ## сводка метрик → reports/SUMMARY.md
	$(PY) scripts/summary.py

all: data train evaluate benchmark loadtest report test  ## полный цикл с нуля

smoke:           ## весь пайплайн на синтетике за 1–2 мин без интернета (проверка кода)
	$(MAKE) smoke-run CONFIG=configs/smoke.yaml

smoke-run:
	$(PY) scripts/dev/make_synthetic.py data/smoke/synthetic.csv 4000
	$(PY) scripts/prepare_data.py --source file --path data/smoke/synthetic.csv
	$(PY) scripts/train_linear.py
	$(PY) scripts/dev/make_tiny_model.py data/smoke/synthetic.csv models/smoke/tiny-bert
	$(PY) scripts/train_transformer.py
	$(PY) scripts/calibrate.py
	$(PY) scripts/evaluate.py --robustness-samples 300
	$(PY) -m pytest

download-models: ## скачать обученные модели из GitHub Release (MODELS_URL=...)
	$(PY) scripts/download_models.py $(MODELS_URL)

package-models:  ## упаковать модели для GitHub Release
	tar -czf models.tar.gz models/linear.joblib models/policy.json models/transformer

docker-build:    ## собрать образ
	docker build -t ugc-moderation .

docker-run:      ## запустить образ с моделями из ./models
	docker run --rm -p 8000:8000 -v $(CURDIR)/models:/app/models:ro ugc-moderation
