from pydantic import BaseModel, Field
from typing import List

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