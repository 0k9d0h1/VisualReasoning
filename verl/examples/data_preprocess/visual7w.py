# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
import base64
from io import BytesIO
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/geo3k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "ruanchaves/visual7w-gpt"

    dataset = datasets.load_dataset(data_source)
    dataset = dataset["train"]
    dataset = dataset.filter(lambda example: example["label"] == "True")

    # Split into new train/test sets (80/20 split)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset  = split_dataset["test"]

    tool_prompt = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.
<tools>
{"type": "function", "function": {"name": "code_executor", "description": "A tool for executing the python codes.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The model's code to be executed, must be a python code"}}, "required": ["code"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

    instruction_following = """You are an AI agent who specializes in visual question answering (VQA).
I want you to reason about how to solve the question.
You can use your image manipulation ability to get the answer right and you can write a code that returns images as output. If you have to write those kind of codes, the code_executor tool will execute the code and show you outputs of that code.


The jupyter notebook has already executed the following code to import the necessary packages:
```python
from PIL import Image
from IPython.display import display
```


REQUIREMENTS #:
1. If you think you got the answer, use Answer: <your answer> to provide the answer.
2. All images in the initial user request are stored in PIL Image objects named image_1, image_2, ..., image_n. You can use these images in your code blocks.
3. There will be no given coordinate information of the image. But, you can find the size of the image via code and use it to crop or zoom the image by approximately estimating the coordinates.
4. You can use display() function to show the image or text output in the notebook for you too see.


Question: """

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("question")
            answer = example.pop("answer")
            image = example.pop("image")
            prompt = instruction_following

            buffered = BytesIO()
            image.save(buffered, format="PNG")
            url = base64.b64encode(buffered.getvalue()).decode("utf-8")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": tool_prompt
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "image": "data:image/png;base64," + url,
                            },
                            {
                                "type": "text",
                                "text": problem + "# USER IMAGE stored in image_1 as PIL image.\nNow please generate only the first <think> and <action> in RESULT. If no action needed, also reply with ANSWER: <your answer> and ends with TERMINATE in the RESULT:\n# RESULT #:\n"
                            }
                        ],
                    }
                ],
                "images": [image],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "code_executor": {
                            "create_kwargs": {"ground_truth": answer},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
