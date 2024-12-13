from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List, TypeVar, Generic, Optional
import json
from loguru import logger as log
from textwrap import dedent
from jinja2 import Template


class LeafTopic(BaseModel):
    name: str = Field(description="The name of the leaf topic")
    description: str = Field(description="A one-sentence description of the leaf topic")

class Category(BaseModel):
    name: str = Field(description="The name of the category")
    description: str = Field(description="A one-sentence description of the category")
    topics: List[str] = Field(description="A list of topics that belong to the category")

class Categorizer_Agent_Output(BaseModel):
    categories: List[Category] = Field(description="A list of categories with the coresponding topics")

class HarmonizerOutput(BaseModel):
    categories: List[Category] = Field(description="A list of categories with the coresponding topics")
    uncategorized_topics: List[LeafTopic] = Field(description="A list of topics that could not be categorized")

class HarmonizerInput(BaseModel):
    existing_categories: List[Category] = Field(description="A list of existing categories.")
    new_topics: List[LeafTopic] = Field(description="A list of topics to be categorized")

class HierarchyBuilderInput(BaseModel):
    categories: List[Category] = Field(description="A list of categories at the base level of the hierarchy")

class HierarchyBuilderOutput(BaseModel):
    hierarchy: List[List[Category]] = Field(description="A list of lists of categories, where each list represents a level in the hierarchy")


InputType = TypeVar('InputType', bound=BaseModel)
OutputType = TypeVar('OutputType', bound=BaseModel)


class LLM_Agent(ABC, Generic[InputType, OutputType]):
    def __init__(self,
                  client: OpenAI,
                  system_instruction: str,
                  response_base_model: BaseModel,
                  model_name: str = "gpt-4o-mini",
                  max_retries: int = 3):
        self.client = client
        self.system_instruction = system_instruction
        self.response_base_model = response_base_model
        self.max_retries = max_retries
        self.model_name = model_name

    def _run(self, user_input: str, input_data: InputType) -> BaseModel:
        prompt = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": user_input}
        ]
        num_retries = 0

        temperature = 0.0

        log.debug(f"Prompt: {prompt}")

        while num_retries < self.max_retries:

            temperature += 0.1 * num_retries

            response = self.client.beta.chat.completions.parse(
                model = self.model_name,
                messages = prompt,
                temperature = temperature,
                seed=42,
                response_format = self.response_base_model
            )
            results_json =  json.loads(response.choices[0].message.content)

            if self._validate_response(input_data, self.response_base_model(**results_json)) is False:
                log.warning(f"Failed Validation in attempt {num_retries + 1}. Retrying...")
                num_retries += 1
            else:
                log.info(f"Successfully validated {self.__class__.__name__} in attempt {num_retries + 1}.")
                return self.response_base_model(**results_json)

        log.error(f"Failed Validation after {self.max_retries} attempts. Ending...")
        return None
    
    @abstractmethod
    async def process(self, input_data: InputType) -> Optional[OutputType]:
        pass
    
    @abstractmethod
    def _validate_response(self, input_data: InputType, response: OutputType) -> bool:
        pass



class Categorizer_Agent(LLM_Agent[List[LeafTopic], Categorizer_Agent_Output]):
    def __init__(self, client: OpenAI, system_instruction: str = None,
                 response_base_model: BaseModel = Categorizer_Agent_Output, 
                 model_name: str = "gpt-4o-mini", max_retries: int = 3, domain: str = None):
        if system_instruction is None:
            assert domain is not None, "If system_instruction is not provided, domain must be provided."
            system_instruction = dedent(f"""\
                                You are a taxonomy expert in customer reviews. 
                                You are given a list of topics and descriptions from the domain {domain}.
                                Your task is to group these topics into high-level categories that make sense for a customer review analysis.

                                Guidelines:
                                1) Every topic must be included in a category.
                                2) Each category should have a name and a one-sentence description.
                                3) The category names should represent the all of the topics that belong to the category.
                                4) The categories should be distinct from each other.
                                5) The categories can vary in size.
                                
                                Important:
                                - Do not leave any topics uncategorized.
                                - For outlier topics, create a new category for them.""")
                                
        super().__init__(client, system_instruction, response_base_model,
                          model_name = model_name, max_retries = max_retries)

    
    def process(self, topics: List[LeafTopic]) -> BaseModel:
        

        user_input_template = dedent("""\
                            {% for item in items %}
                            {{ item.name }}: {{ item.description }}.
                            {% endfor %}""")
        user_input_template  = Template(user_input_template)
        user_input = user_input_template.render(items=topics)
        return self._run(user_input, input_data=topics)


    def _validate_response(self, input_data: List[LeafTopic], response: Categorizer_Agent_Output) -> bool:
        # Check if all topics are included in a category
        input_topics = [topic.name for topic in input_data]
        all_topics_included = all(topic in [t for category in response.categories for t in category.topics] for topic in input_topics)

        # Check if all categories have a non-empty name and a description
        all_categories_named = all(category.name and category.description for category in response.categories)

        # Check if the categories are distinct from each other
        category_names = [category.name for category in response.categories]
        categories_distinct = len(category_names) == len(set(category_names))

        if not all_topics_included:
            log.warning("Not all topics were included in a category by the Categorizer Agent. Will be re-tried if attempts remain.")
            misssing_topics = [topic for topic in input_topics if topic not in [t for category in response.categories for t in category.topics]]
            log.warning(f"Missing topics: {misssing_topics}")
        if not all_categories_named:
            log.warning("Not all categories have a non-empty name and a description. Will be re-tried if attempts remain.")
        if not categories_distinct:
            log.warning("Categories are not distinct from each other. Will be re-tried if attempts remain.")

        return all_topics_included and all_categories_named and categories_distinct


class CategoryHarmonizerAgent(LLM_Agent[HarmonizerInput, HarmonizerOutput]):
    def __init__(self, client: OpenAI, system_instruction: str = None, response_base_model: BaseModel = HarmonizerOutput, domain: str = None):
        if system_instruction is None:
            assert domain is not None, "If system_instruction is not provided, domain must be provided."
            system_instruction = dedent(f"""\
                                You are a taxonomy expert in customer reviews. Given:
                                1) A list of existing high-level categories and their descriptions from the domain {domain}.
                                2) A new batch of topics and their descriptions from the same domain.

                                Your task is to:
                                1) Evaluate each topic against existing categories.
                                2) Assign topics to categories where there's a clear fit
                                3) Flag topics that don't fit well anywhere.
                                4) All topics must be included in a category or flagged as uncategorized.

                                Do not create new categories.""")
        super().__init__(client, system_instruction, response_base_model)
    
    def process(self, input_data: HarmonizerInput) -> HarmonizerOutput:
        user_input_template = dedent("""\
                            Existing categories:
                            {% for category in existing_categories %}
                            {{ category.name }}: {{ category.description }}.
                            {% endfor %}

                            New topics:
                            {% for topic in new_topics %}
                            {{ topic.name }}: {{ topic.description }}.
                            {% endfor %}""")
        user_input_template  = Template(user_input_template)
        user_input = user_input_template.render(existing_categories=input_data.existing_categories, new_topics=input_data.new_topics)
        return self._run(user_input, input_data=input_data)
    
    def _validate_response(self, input_data: HarmonizerInput, response: HarmonizerOutput) -> bool:
        # 1) All topics must be included in a category or flagged as uncategorized.
        input_topics = [topic.name for topic in input_data.new_topics]
        all_topics_included = all(topic in [t for category in response.categories for t in category.topics] 
                                  + [t.name for t in response.uncategorized_topics] for topic in input_topics)

        # 2) No new categories should be created.
        existing_category_names = [category.name for category in input_data.existing_categories]
        new_category_names = [category.name for category in response.categories]
        no_new_categories = all(name in existing_category_names for name in new_category_names)


        if not all_topics_included:
            # log a list of missing topics
            missing_topics = [topic for topic in input_topics if topic not in [t for category in response.categories for t in category.topics] + response.uncategorized_topics]
            log.warning(f"The following topics were not included in a category or flagged as uncategorized: {missing_topics}. Will be re-tried if attempts remain.")
        if not no_new_categories:
            log.warning(f"Existing categories: {existing_category_names}")
            log.warning(f"New categories: {new_category_names}")
            log.warning("New categories were created by the Category Harmonizer Agent. Will be re-tried if attempts remain.")

        return all_topics_included and no_new_categories

class HierarchyBuilder(LLM_Agent[HierarchyBuilderInput, HierarchyBuilderOutput]):
    def __init__(self, client: OpenAI, 
                system_instruction: str = None,
                response_base_model: BaseModel = HierarchyBuilderOutput, 
                domain: str = None,
                max_depth: int = 3):
        if system_instruction is None:
            assert domain is not None, "If system_instruction is not provided, domain must be provided."
            system_instruction = dedent(f"""\
                                        You are a taxonomy expert in customer reviews. Given a list of topics and their descriptions from the domain {domain}:
                                        1) Construct a {max_depth}-level hierarchical clustering of these topics.
                                        2) Every node in the hierarchy should hava a name and a one-sentence description.
                                        3) Each topic can only belong to one parent category""")
        super().__init__(client, system_instruction, response_base_model)

    def process(self, input_data: HierarchyBuilderInput) -> HierarchyBuilderOutput:
        user_input_template = dedent("""\
                                      {% for category in categories %}
                                      {{ category.name }}: {{ category.description }}.
                                      {% endfor %}""")
        user_input_template  = Template(user_input_template)
        user_input = user_input_template.render(categories=input_data.categories)
        return self._run(user_input, input_data=input_data)

    def _validate_response(self, input_data: HierarchyBuilderInput, response: HierarchyBuilderOutput) -> bool:
        return True


        

class TaxonomyBuilder:
    def __init__(self, client: OpenAI, domain: str):
        self.client = client
        self.domain = domain
        self.categorizer_agent = CategorizerAgent(client, domain=domain)
        self.harmonizer_agent = CategoryHarmonizerAgent(client, domain=domain)

    def build_taxonomy(self, topics: List[LeafTopic], max_attempts: int = 3) -> List[Category]:
        for attempt in range(max_attempts):
            log.info(f"Attempt {attempt + 1} to build taxonomy.")
            categorized_topics = self.categorizer_agent.process(topics)
            if self.categorizer_agent._validate_response(topics, categorized_topics):
                return categorized_topics.categories
            else:
                log.info("Categorizer Agent's output did not pass validation. Will try again if attempts remain.")

        log.warning("Categorizer Agent failed to produce a valid output after multiple attempts. Proceeding with harmonization.")
        harmonized_topics = self.harmonizer_agent.process(HarmonizerInput(existing_categories=[], new_topics=topics))
        if self.harmonizer_agent._validate_response(HarmonizerInput(existing_categories=[], new_topics=topics), harmonized_topics):
            return harmonized_topics.categories
        else:
            log.error("Category Harmonizer Agent also failed to produce a valid output. Please check the input data or the agents' configurations.")
            return []

        
    
