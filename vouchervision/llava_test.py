from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.eval.run_llava import eval_model

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

# model_path = "liuhaotian/llava-v1.5-7b"
# model_path = "liuhaotian/llava-v1.6-mistral-7b"
model_path = "liuhaotian/llava-v1.6-34b"
prompt = """I need you to transcribe all of the text in this image. Place the transcribed text into a JSON dictionary with this form {"Transcription": "text"}"""
# image_file = "https://llava-vl.github.io/static/images/view.jpg"
image_file = "/home/brlab/Dropbox/VoucherVision/demo/demo_images/MICH_16205594_Poaceae_Jouvea_pilosa.jpg"
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    # "load_8_bit": True,
})()

eval_model(args)