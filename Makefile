.PHONY: prepare-data train evaluate predict test

prepare-data:
	python -m weather_korea_forecast.data.build_training_table --config configs/data/dataset_v1.yaml

train:
	python -m weather_korea_forecast.training.train --data-config configs/data/dataset_v1.yaml --model-config configs/model/tft_v1.yaml --train-config configs/train/train_v1.yaml

evaluate:
	python -m weather_korea_forecast.evaluation.evaluate --experiment-dir data/artifacts/experiments/latest

predict:
	python -m weather_korea_forecast.inference.predict --experiment-dir data/artifacts/experiments/latest --station-id SEOUL --forecast-init-time 2025-01-03T00:00:00Z

test:
	pytest -q
