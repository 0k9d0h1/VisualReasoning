# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import json
import logging
import os
from io import BytesIO
import glob
import base64
from PIL import Image
from typing import Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score import gsm8k

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

from autogen.coding import CodeBlock
from autogen.coding.jupyter import JupyterCodeExecutor, LocalJupyterServer
import ast, re

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CodeExecutionTool(BaseTool):
    """A tool for executing python codes.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_execution_tool",
                "description": "A tool for executing python codes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to be executed",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        self.working_dir = config.get("working_dir", "")
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)

        # set up the server
        self.server = LocalJupyterServer()
            
        # set up the jupyter executor
        self.executor = JupyterCodeExecutor(self.server, output_dir=self.working_dir, kernel_name="sketchpad")
        
        # initialize the environment
        self.init_env()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: str, **kwargs) -> Tuple[str, float, dict]:
        try:
            _parameters = json.loads(parameters)
        except json.JSONDecodeError:
            _parameters = {}
        if isinstance(_parameters, dict):
            code = _parameters.get("code", "")
        else:
            code = ""
        self._instance_dict[instance_id]["response"] = code

        self.executor._jupyter_kernel_client = self.executor._jupyter_client.get_kernel_client(self.executor._kernel_id)
        execution_result = self.executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python",
                        code=code),
            ]
        )
        ret = await self.result_processor(execution_result)

        reward = await self.calc_reward(instance_id)
        # penalty for non improved answer submission
        tool_reward = 0.0
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward
        return ret, tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0
    
    async def result_processor(self, result):
        # Change an IPythonCodeResult object to a string, and the list of files
        # If the execution failed, the string is the error message.
        # If the execution is successful, the string is the output of the code execution.
        # In the string, all embeded PIL images are replaced by their file paths, using html img tag.
        # The list of files are the paths of the images.
        
        # process error message
        async def parse_error_message(error):
            # Find the index where the list starts, indicated by `['`
            list_start_index = error.find("['")
            
            # The first part before the list is the initial error message
            initial_error = error[:list_start_index].strip()
            
            # The second part is the list of strings, which starts from `['` and goes to the end of the string
            traceback_list_str = error[list_start_index:]
            
            # Use ast.literal_eval to safely evaluate the string representation of the list
            # This method is safer than eval and can handle Python literals properly
            try:
                traceback_list = ast.literal_eval(traceback_list_str)
            except SyntaxError as e:
                print("Error parsing the list: ", e)
                traceback_list = []
                
            # Remove ANSI escape sequences
            ansi_escape = re.compile(r'\x1b\[.*?m')
            traceback_list = [ansi_escape.sub('', line) for line in traceback_list]
            
            return initial_error + "\n\n" + "\n".join(traceback_list)
        
        
        exit_code = result.exit_code
        
        file_paths = result.output_files
        output_str = result.output
        output_lines = output_str.split("\n")
        
        if len(file_paths) > 0:
            output_lines = output_lines[:-2*len(file_paths)]

        output_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<tool_response>\n"
                }
            ]
        }
            
        # if execution succeeded, replace PIL images with their file paths
        if exit_code == 0:
            new_str = ""
            image_idx = 0
            
            for line in output_lines:
                if line.startswith("<PIL."):
                    if image_idx < len(file_paths):
                        image = Image.open(file_paths[image_idx])
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        output_message["content"].append(
                            {
                                "type": "image",
                                "image": "data:image/png;base64," + image_base64
                            }
                        )
                        image_idx += 1
                else:
                    output_message["content"].append(
                            {
                                "type": "text",
                                "text": "data:image/png;base64," + line
                            }
                        )
                new_str += "\n"
            
            # add the remaining images
            for file_idx, file in enumerate(file_paths):
                if file_idx >= image_idx:
                    image = Image.open(file)
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    output_message["content"].append(
                        {
                            "type": "image",
                            "image": "data:image/png;base64," + image_base64
                        }
                    )
            
            output_message["content"].append(
                {
                    "type": "text",
                    "text": "</tool_response>\n"
                }
            )
                
            return exit_code, output_message, file_paths
        
        # if execution failed, parse the error message
        else:
            error_msg = await parse_error_message(output_str)
            output_message["content"].append(
                {
                    "type": "text",
                    "text": error_msg + "</tool_response>\n"
                }
            )
            return exit_code, output_message, file_paths

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

    def init_env(self):
        init_code = ("import sys\n"
                     "from PIL import Image\n"
                     "from IPython.display import display\n"
        )
        
        self.executor._jupyter_kernel_client = self.executor._jupyter_client.get_kernel_client(self.executor._kernel_id)
        self.executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python",
                        code=init_code),
            ]
        )


    async def cleanup(self):
        self.server.stop()

