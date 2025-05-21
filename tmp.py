from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd


processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# Load the data
data_path = "/home/kdh0901/Desktop/Train_VisualReasoning/data/visual7w_train.parquet"
data = pd.read_parquet(data_path)

# Process each "messages" column entry with the processor
for text in data["messages"]:
    print(text)
    processor.apply_chat_template(text, tokenize=True, return_tensors="pt", add_generation_prompt=False)