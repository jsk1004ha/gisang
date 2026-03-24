#!/usr/bin/env bash
python -m weather_korea_forecast.training.train --data-config configs/data/dataset_v1.yaml --model-config configs/model/tft_v1.yaml --train-config configs/train/train_v1.yaml
