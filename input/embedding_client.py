import numpy as np
import ollama
import asyncio
import logging


class OllamaEmbeddingClient:
    """Client for Ollama embedding using the official Python client"""
    
    def __init__(self,logger):
        self.client = ollama.Client()
        self.logger = logger
    
    def get_embedding(self, model: str, text: str) -> np.ndarray:
        """Get embedding from Ollama using the official client"""
        try:
            response = self.client.embed(model=model, input=text)
            if 'embeddings' in response and len(response['embeddings']) > 0:
                return np.array(response['embeddings'][0])
            else:
                raise ValueError(f"No embeddings returned from model {model}")
        except Exception as e:
            self.logger.error(f"Error getting embedding from {model}: {e}")
            raise
    
    async def get_embedding_async(self, model: str, text: str) -> np.ndarray:
        """Async wrapper for getting embeddings"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, model, text)