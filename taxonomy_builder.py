from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List, TypeVar, Generic, Optional


class LeafTopic(BaseModel):
    name: str = Field(description="The name of the leaf topic")
    description: str = Field(description="A one-sentence description of the leaf topic")

class Category(BaseModel):
    name: str = Field(description="The name of the category")
    description: str = Field(description="A one-sentence description of the category")
    topics: List[str] = Field(description="A list of topics that belong to the category")

class Categorizer_Agent_Output(BaseModel):
    categories: List[Category] = Field(description="A list of categories with the coresponding topics")

class Category_Harmonizer_Agent_Output(BaseModel):
    categories: List[Category] = Field(description="A list of categories with the coresponding topics")
    uncategorized_topics: List[LeafTopic] = Field(description="A list of topics that could not be categorized")

InputType = TypeVar('InputType', bound=BaseModel)
OutputType = TypeVar('OutputType', bound=BaseModel)

class LLM_Agent(ABC, Generic[InputType, OutputType]):
    def __init__(self,
                  client: OpenAI,
                  system_instruction: str,
                  response_base_model: BaseModel,
                  max_retries: int = 3):
        self.client = client
        self.system_instruction = system_instruction
        self.response_base_model = response_base_model
        self.max_retries = max_retries

    def _run(self, user_input: str) -> BaseModel:
        prompt = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": user_input}
        ]
        num_retries = 0
        while num_retries < self.max_retries:

            response = self.client.beta.chat.completions.parse(
                model = "gpt-4o-mini",
                messages = prompt,
                temperature = 0.0,
                seed=42,
                response_format = self.response_base_model
            )
            results_json =  json.loads(response.choices[0].message.content)
            if self._validate_response(self.response_base_model(**results_json)) is False:
                log.warning(f"Failed topic validation in attempt {num_retries + 1}. Retrying...")
                num_retries += 1
            else:
                return self.response_base_model(**results_json)
        log.error(f"Failed topic validation after {self.max_retries} attempts. Skipping...")
        return None
    
    @abstractmethod
    async def process(self, input_data: InputType) -> Optional[OutputType]:
        pass
    
    def _validate_response(self, response: BaseModel) -> bool:
        raise NotImplementedError
