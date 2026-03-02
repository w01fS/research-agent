import subprocess

class OllamaClient:
    """
    Lightweight local LLM client wrapper for Ollama models.
    Example:
        client = OllamaClient("llama3.2:3b")
        output = client.generate(prompt)
    """

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Calls the local Ollama model via CLI using stdin to pass the prompt.
        This mirrors the previous working `call_llm` behavior.
        """
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if result.returncode != 0:
                # Return stderr so parser sees the error if model fails
                return f"LLM execution error: {result.stderr.decode().strip()}"

            return result.stdout.decode().strip()

        except Exception as e:
            return f"LLM execution exception: {str(e)}"