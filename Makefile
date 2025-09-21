all:

run:
	poetry run python client_llama_cpp.py
	
install_gpu:
	rm -rf .venv
	poetry install
	CMAKE_ARGS="-DLLAVA_BUILD=OFF -DGGML_CUDA=ON" poetry run python -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
