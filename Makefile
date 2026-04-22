.PHONY: train eval multi-seed test lint fmt clean

train:
	python scripts/train.py --config configs/resnet20.yaml

multi-seed:
	python scripts/multi_seed.py --config configs/resnet20.yaml

eval:
	@[ "$(CHECKPOINT)" ] || { echo "Usage: make eval CHECKPOINT=runs/.../best.pth"; exit 1; }
	python scripts/evaluate.py --checkpoint $(CHECKPOINT)

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff format .

clean:
	rm -rf runs/ artifacts/*.png artifacts/*.json __pycache__ .pytest_cache .ruff_cache
