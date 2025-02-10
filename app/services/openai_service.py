
from typing import List
from app.core.config import settings
from app.core.logging import logger
from openai import OpenAI

class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            return [r.embedding for r in response.data]
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise