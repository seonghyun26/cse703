train: PPO_jssp_multiInstances.py
	python3 PPO_jssp_multiInstances.py

test:
	python3 test_learned.py

test_bench:
	python3 test_learned_on_benchmark.py