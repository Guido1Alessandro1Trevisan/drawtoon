import json
from typing import Dict, List, Tuple

import torch
import transformers
from tqdm import tqdm

from layousyn.prompt_handler.cache import KeyObjectCache
from layousyn.prompt_handler.common import LABEL_SET_GENERATION_PROMPT, get_llama_gen_pipeline

from .base import BasePromptHandler, PromptInfo


@torch.no_grad()
def process_sentences(
    prompts: List[str], pipeline: transformers.Pipeline, terminators: List[int]
) -> List[List[Tuple[str, int]]]:
    outputs = pipeline(
        prompts,
        max_new_tokens=60,
        eos_token_id=terminators,
        do_sample=False,
        top_p=None,
        temperature=None,
        pad_token_id=pipeline.tokenizer.eos_token_id,
    )

    results = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        output = outputs[i][0]["generated_text"][len(prompt):].lower()
        # Strip Qwen3 thinking block if present
        if "</think>" in output:
            output = output.split("</think>")[-1].strip()
        try:
            output_json = json.loads(output)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {output}")
            output_json = {}
        results.append([(obj, count) for obj, count in output_json.items()])

    return results


class QwenObjectCountPromptHandler(BasePromptHandler):
    def __init__(
        self,
        model_id="Qwen/Qwen3-8B",
        batch_size=1,
        device="cuda",
        cache_path=None,
    ):
        super().__init__()
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device

        self._pipeline = None
        self._tokenizer = None
        self._terminators = None

        if cache_path is None:
            cache_path = "cache/" + model_id.replace("/", "_").replace("-", "_") + "_cache.db"
        self.cache = KeyObjectCache(cache_path)

    def _lazy_init(self):
        if self._pipeline is None:
            self._pipeline = get_llama_gen_pipeline(
                self.model_id,
                batch_size=self.batch_size,
                device=self.device,
            )
            self._tokenizer = self._pipeline.tokenizer
            self._terminators = [self._pipeline.tokenizer.eos_token_id]

    @property
    def pipeline(self):
        self._lazy_init()
        return self._pipeline

    @property
    def tokenizer(self):
        self._lazy_init()
        return self._tokenizer

    @property
    def terminators(self):
        self._lazy_init()
        return self._terminators

    @staticmethod
    def get_modified_prompt(sentence: str, tokenizer) -> str:
        messages = [
            {"role": "system", "content": LABEL_SET_GENERATION_PROMPT},
            {"role": "user", "content": f"{sentence}\nPlausible scene: "},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return prompt

    def get_info(self, prompts: List[str]) -> List[PromptInfo]:
        prompt_data_dict = self.cache.get_key_object_map(prompts)
        cached_results: Dict[str, PromptInfo] = {}
        for prompt, prompt_data in prompt_data_dict.items():
            cached_results[prompt] = PromptInfo.from_json(prompt_data)

        prompts_to_process = list(set(prompts) - set(cached_results.keys()))

        if not prompts_to_process:
            return [cached_results[prompt] for prompt in prompts]

        print(f"Processing {len(prompts_to_process)} prompts")

        new_results = {}
        for i in tqdm(range(0, len(prompts_to_process), self.batch_size), desc="Processing prompts"):
            batch_prompts = prompts_to_process[i: i + self.batch_size]
            modified_batch_prompts = [
                QwenObjectCountPromptHandler.get_modified_prompt(prompt, self.tokenizer)
                for prompt in batch_prompts
            ]
            batch_outputs = process_sentences(modified_batch_prompts, self.pipeline, self.terminators)
            for prompt, output in zip(batch_prompts, batch_outputs):
                prompt_info = PromptInfo(
                    objects=set(obj for obj, _ in output),
                    objectWithCounts={obj: count for obj, count in output},
                )
                new_results[prompt] = prompt_info
                self.cache.insert(prompt, prompt_info.to_json())

        return [new_results.get(prompt) or cached_results.get(prompt) or PromptInfo(objects=set(), objectWithCounts={}) for prompt in prompts]
