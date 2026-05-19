import json
from typing import Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm

from layousyn.prompt_handler.cache import KeyObjectCache
from layousyn.prompt_handler.common import LABEL_SET_GENERATION_PROMPT

from .base import BasePromptHandler, PromptInfo


class GPTObjectCountPromptHandler(BasePromptHandler):
    def __init__(
        self,
        model_id="gpt-3.5-turbo",
        batch_size=1,
        cache_path=None,
        **kwargs,
    ):
        super().__init__()
        self.pipeline = OpenAI()
        self.model_id = model_id
        self.batch_size = batch_size

        # initialize the cache
        if cache_path is None:
            cache_path = (
                "cache/" + model_id.replace("/", "_").replace("-", "_") + "_cache.db"
            )
        self.cache = KeyObjectCache(cache_path)

    @staticmethod
    def process_sentences(
        messages: List[List[Dict[str, str]]],
        pipeline: OpenAI,
        model_id: str,
    ) -> List[List[Tuple[str, int]]]:
        """
        Process the given sentences with the given pipeline.

        Args:
            sentences (List[str]): The sentences to process.
            pipeline (transformers.Pipeline): The pipeline to use.
        """

        results = []
        for i in range(len(messages)):
            completion = pipeline.chat.completions.create(
                model=model_id, messages=messages[i]
            )  # type: ignore
            output = completion.choices[0].message.content.lower()  # type: ignore
            try:
                output_json = json.loads(output)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {output}")
                output_json = {}
            results.append([(obj, count) for obj, count in output_json.items()])

        return results

    @staticmethod
    def get_modified_prompt(sentence: str) -> List[Dict[str, str]]:
        """
        Generate a modified prompt for the given sentence.

        Args:
            sentence (str): The sentence to generate a prompt for.
            pipeline (transformers.Pipeline): The pipeline to use.

        Returns:
            str: The generated prompt.
        """
        messages = [
            {
                "role": "system",
                "content": LABEL_SET_GENERATION_PROMPT,
            },
            {"role": "user", "content": f"{sentence}\nPlausible scene: "},
        ]

        return messages

    def get_info(self, prompts: List[str]) -> List[PromptInfo]:
        # check the cache
        prompt_data_dict = self.cache.get_key_object_map(prompts)  # type: ignore
        cached_results: Dict[str, PromptInfo] = {}
        for prompt, prompt_data in prompt_data_dict.items():
            cached_results[prompt] = PromptInfo.from_json(prompt_data)

        # get the prompts that are not in the cache
        prompts_to_process = set(prompts) - set(cached_results.keys())
        prompts_to_process = list(prompts_to_process)

        if not prompts_to_process:
            return [cached_results[prompt] for prompt in prompts]

        # process the remaining prompts
        print(f"Processing {len(prompts_to_process)} prompts")

        # process in batches
        new_results = {}
        for i in tqdm(
            range(0, len(prompts_to_process), self.batch_size),
            desc="Processing prompts",
        ):
            batch_prompts = prompts_to_process[i : i + self.batch_size]
            modified_batch_prompts = [
                GPTObjectCountPromptHandler.get_modified_prompt(prompt)
                for prompt in batch_prompts
            ]
            batch_outputs = GPTObjectCountPromptHandler.process_sentences(
                modified_batch_prompts, self.pipeline, self.model_id  # type: ignore
            )
            for prompt, output in zip(batch_prompts, batch_outputs):
                objects = set([obj for obj, _ in output])
                objectWithCounts = {obj: count for obj, count in output}
                prompt_info = PromptInfo(
                    objects=objects,
                    objectWithCounts=objectWithCounts,
                )

                # update the results
                new_results[prompt] = prompt_info

                # update the cache
                self.cache.insert(prompt, prompt_info.to_json())

        # combine the results
        results = []
        for prompt in prompts:
            if prompt in new_results:
                results.append(new_results[prompt])
            else:
                results.append(cached_results[prompt])

        return results
