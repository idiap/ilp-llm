#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Joao Pedro <joao.gandarela@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def clean_answer(answer: str):
    answer = answer.replace("user", "").replace("assistant", "").replace("model", "").strip()
    answer = answer.replace("#My", "").replace("The output is", "")
    answer = answer.replace("Answer", "").replace("The answer is", "")
    answer = answer.replace("[INST]", "").replace("[/INST]", "")
    answer = answer.replace("<", "").replace(">", "")
    answer = answer.strip()

    return answer

class LocalLLM:
    def __init__(self, model_locator: str, logging: bool = False):
        self.model_locator = model_locator
        self.tokenizer = AutoTokenizer.from_pretrained(model_locator, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_locator, device_map="auto", torch_dtype="auto",
                                                          attn_implementation="flash_attention_2")

        if (not self.tokenizer.pad_token):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logging = logging

    def prompt(self, messages, output_size: int) -> str:
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(inputs)
        tokens = self.tokenizer(inputs, padding=True, return_tensors="pt")
        generated = self.model.generate(tokens.input_ids[:, -8192:].to(self.model.device),
                                        attention_mask=tokens.attention_mask[:, -8192:].to(self.model.device),
                                        max_new_tokens=output_size)

        response = generated[0][tokens.input_ids.shape[-1]:]

        decoded = self.tokenizer.decode(response, skip_special_tokens=True)

        # print(clean_answer(decoded))
        # exit(0)

        return clean_answer(decoded)
