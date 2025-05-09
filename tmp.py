from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info


processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR4c1y_7gy2crn2Ll_ZSWzcqb0WDZFuBnFTeQ&s"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": 
         [   {"type": "image", "image": "https://img.freepik.com/free-vector/set-black-irregular-blobs-random-liquid-uneven-drop-shape-amorphous-splodges-asymmetric-spots_88138-2073.jpg?semt=ais_hybrid&w=740"},
            {"type": "text", "text": "What are the common elements in these pictures?"},]},
]
# Combine messages for batch processing
messages = [messages1, messages2]

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
print(texts)
image_inputs, video_inputs = process_vision_info(messages)
print(processor.tokenizer("<|im_start|>"))
print(processor.tokenizer("assistant"))
print(processor.tokenizer("<|im_end|>"))
print(image_inputs)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
cnt = 0
for t in inputs.input_ids:
    cnt += t.tolist().count(151655)
print(cnt)
print(inputs.pixel_values.shape)
inputs = inputs.to("cuda")

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
text1 = [processor.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)]
text2 = [processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)]
image_inputs1, video_inputs1 = process_vision_info(messages1)
image_inputs2, video_inputs2 = process_vision_info(messages2)
inputs1 = processor(
    text=text1,
    images=image_inputs1,
    videos=video_inputs1,
    padding=True,
    return_tensors="pt",
)
inputs2 = processor(
    text=text2,
    images=image_inputs2,
    videos=video_inputs2,
    padding=True,
    return_tensors="pt",
)

print(inputs1.input_ids.shape)

cnt1 = 0
cnt2 = 0
for t in inputs1.input_ids:
    cnt1 += t.tolist().count(151655)
for t in inputs2.input_ids:
    cnt2 += t.tolist().count(151655)
print(cnt1)
print(cnt2)
print(inputs1.pixel_values.shape)
print(inputs2.pixel_values.shape)