from ollama import chat, ChatResponse

class OllamaChatClient:
    def __init__(self, logger, model_name: str = "gemma3:1b"):
        self.model_name = model_name
        self.logger = logger

    def send_prompt(self, prompt: str) -> str:
        try:
            response: ChatResponse = chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response.message.content  # Preferred
        except Exception as e:
            self.logger.error(f"❌ Error during single prompt prediction: {e}")
            return f"⚠️ Error: {str(e)}"
            
        