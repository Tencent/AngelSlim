# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""


import dataclasses

import numpy as np

from .instruction_following_eval import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def read_prompt_list(data):
    """Read input`s from data."""
    inputs = []

    for example in data:
        inputs.append(
            InputExample(key=example['key'],
                         instruction_id_list=example['instruction_id_list'],
                         prompt=example['prompt'],
                         kwargs=example['kwargs']))
    return inputs


def test_instruction_following_strict(
        inp,
        prompt_to_response,
):
    """Tests response to see if instrutions are followed."""
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and 'prompt' in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
        inp,
        prompt_to_response,
):
    """Tests response for an upper bound for following instructions."""
    response = prompt_to_response[inp.prompt]
    r = response.split('\n')
    response_remove_first = '\n'.join(r[1:]).strip()
    response_remove_last = '\n'.join(r[:-1]).strip()
    response_remove_both = '\n'.join(r[1:-1]).strip()
    revised_response = response.replace('*', '')
    revised_response_remove_first = response_remove_first.replace('*', '')
    revised_response_remove_last = response_remove_last.replace('*', '')
    revised_response_remove_both = response_remove_both.replace('*', '')
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and 'prompt' in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def read_prompt_to_response_dict(data):
    """Creates dictionary matching prompt and response."""
    return_dict = {}
    for example in data:
        if isinstance(example, list):
            example = example[0]
        tmp = example['response']
        if tmp.startswith('<1>') or tmp.startswith('<2>') or tmp.startswith('<3>'):
            tmp = tmp[3:].strip()
        if tmp.endswith('<|user|>'):
            tmp = tmp[:-8].strip()

        return_dict[example['prompt']] = tmp
    return return_dict


def print_report(outputs):
    """Prints a report on accuracy scores."""

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)
    return {
        'prompt': prompt_correct / prompt_total,
        'instruction': instruction_correct / instruction_total
    }


def evaluate(inputs, prompt_to_response):
    # get instruction following results
    results = {}
    outputs = []
    for inp in inputs:
        outputs.append(test_instruction_following_strict(
            inp, prompt_to_response))
    for key, value in print_report(outputs).items():
        results[f'strict-{key}'] = value
    outputs = []
    for inp in inputs:
        outputs.append(test_instruction_following_loose(
            inp, prompt_to_response))
    for key, value in print_report(outputs).items():
        results[f'loose-{key}'] = value
    results['final'] = np.mean(list(results.values()))
    return results
