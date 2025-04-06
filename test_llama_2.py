from llama_cpp import Llama

model = Llama(
      model_path="/home/spark/dev/models/mtext-141024_mistral-7B-v0.1_merged-Q8-2.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
#prompt = open("/home/spark/dev/prompts/IA-2.txt").read()
prompt = "<|s|>\nHAMLET\nEtre ou ne pas être c'est"
print("PROMPT //"+prompt+"//")

output = model.create_completion(
      prompt, # Prompt
      max_tokens=50, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["<|e|>"], # Stop generating just before the model would generate a new question
      echo=False, # Echo the prompt back in the output
      stream=False
    ) # Generate a completion, can also call create_completion
#prompt += output["choices"][0]["text"]
print("GEN //"+output["choices"][0]["text"]+"//")


