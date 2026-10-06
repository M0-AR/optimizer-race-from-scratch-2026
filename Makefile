.PHONY: quick full docker-quick docker-full media referee spectramix pilot

quick:
	python3 experiments/run_all.py --quick

full:
	python3 experiments/run_all.py --full

docker-quick:
	docker compose run --rm optimizer-race-quick

docker-full:
	docker compose run --rm optimizer-race

media:
	python3 experiments/make_media.py

spectramix:
	python3 experiments/try_spectramix.py

referee:
	python3 experiments/try_referee.py

pilot:
	python3 experiments/try_pilot.py
