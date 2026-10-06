.PHONY: quick full docker-quick docker-full

quick:
	python3 experiments/run_all.py --quick

full:
	python3 experiments/run_all.py --full

docker-quick:
	docker compose run --rm optimizer-race-quick

docker-full:
	docker compose run --rm optimizer-race
