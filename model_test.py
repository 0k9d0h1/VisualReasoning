import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

# Load the Qwen 2.5 VL base model and processor
model_path = '/home/kdh0901/Desktop/cache_dir/kdh0901/checkpoints/global_step_260'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True).to("cuda:0")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)

# Load the first sample from the Visual7W-GPT train split
dataset = load_dataset('hiyouga/geometry3k', split='train')
sample = dataset[19]

# Prepare the image and question
image = sample['images'][0]
question = sample['problem']

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "You are an AI agent who specializes on image manipulation with python code and solving image deduction problems. I want you to reason about how to solve the # USER REQUEST # and generate the actions step by step(each action is a python code enclosed by ```python``` to manipulate the images) to solve the request.\nYou MUST use your image manipulation ability to get the answer right and you can write a code that return images as output. If you have to write those kind of codes, I can execute the code and show you outputs of that code.\n\n\nThe jupyter notebook has already executed the following code to import the necessary packages:\n```python\nfrom PIL import Image\nfrom IPython.display import display\n```\n\n\nREQUIREMENTS #:\n1. The generated actions can resolve the given user request # USER REQUEST # perfectly. The user request is reasonable and can be solved. Try your best to solve the request.\n2. If you think you got the answer, use ANSWER: <your answer> to provide the answer, and ends with TERMINATE.\n3. All images in the initial user request are stored in PIL Image objects named image_1, image_2, ..., image_n. You can use these images in your code blocks. Use display() function to show the image or text output in the notebook for you too see.\n4. If you know the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next <think> and <action>.\n5. There will be no given coordinate information of the image. But, you can find the size of the image via code and use it to crop or zoom the image by approximately estimating the coordinates.\n6. The python codes shoule be enclosed by ```python```.\n7. You can use display() function to show the image or text output in the notebook for you too see.\n\n\nBelow are examples of what format you should generate the outputs in.:\n\n\n# EXAMPLE:\n# USER REQUEST #: <An image here> <question>\n# USER IMAGE stored in image_1, as a PIL image.\n# RESULT #:\n<think>\n<Your reasoning process>\n</think>\n<action>\n```python\nimport cv2\nimport numpy as np\n\n<Python code>\n```\n</action>\n<observation>\nExecution success. The output is as follows:\n<outputs of the previous code are here.>\nIf you can get the answer, please reply with ANSWER: <your answer> and ends with TERMINATE. Otherwise, please generate the next THOUGHT and ACTION.\n</observation>\n<think>\n<Your reasoning process>\n</think>\n<action>\nNo action needed.\n</action>\nANSWER: <your answer>. TERMINATE\n\n\n# USER REQUEST #: " + question},
            {
                "type": "image",
                "image": image,
            },
        ],
    },
    
    {
        "role": "tool",
        "content": [
            {"type": "text", "text": "afsdsdfasdaf " + question},
            {
                "type": "image",
                "image": image,
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
print(image_inputs)
inputs = inputs.to(model.device)
# Inference: sample 10 outputs
answers = []
with torch.no_grad():
    while True:
        outputs = model.generate(
            **inputs,
            max_new_tokens=65536,
            do_sample=True,
            top_p=0.95,
            temperature=0.6,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])
        if "<action>\n```python\n" in output_text[0]:
            print(output_text[0])
            break
        torch.cuda.empty_cache()