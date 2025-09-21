all:

run:
	poetry run python client_llama_cpp.py
	
install_gpu:
	rm -rf .venv
	FORCE_CUDA=1 poetry install
