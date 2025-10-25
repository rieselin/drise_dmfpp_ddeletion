from PIL import Image

from unsloth import FastVisionModel # FastLanguageModel for LLMs
from transformers import TextStreamer

class LLMAVisionModel:
    def __init__(self, args):
        self.args = args
    def load_model(self):
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Llama-3.2-11B-Vision-Instruct",
            load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        )
        FastVisionModel.for_inference(model) # Enable for inference!
        return model, tokenizer
    def createImageInput(self, output_path, resized_img, target_class):
        #yolo detected bboxes
        yoloPredictedBboxes = Image.open(f'{output_path}yolo_predicted_bboxes.png')

        # saliency map with target bbox
        driseSaliency = Image.open(f'{output_path}drise_saliency.png')


        images = [resized_img, yoloPredictedBboxes, driseSaliency]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        composed = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            composed.paste(im, (x_offset,0))
            x_offset += im.size[0]

        composed.save(f'{output_path}composed_image.jpg')
        return composed
    def compose_input(self, tokenizer, composed):
        messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self.args.instruction}
        ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = tokenizer(
            composed,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        return inputs
    def generate_response(self, inputs, tokenizer, model, output_path):
        output_path = f'{output_path}llama_response.txt'
        text_streamer = FileTextStreamer(tokenizer, file_path=output_path)
        _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 400,
                        use_cache = True, temperature = 1.5, min_p = 0.1)
        
        
        
class FileTextStreamer(TextStreamer):
    def __init__(self, tokenizer, file_path, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True, **kwargs)
        self.file = open(file_path, "w", encoding="utf-8")

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Write the generated text chunk to file
        self.file.write(text)
        self.file.flush()  # ensure content is written immediately
        if stream_end:
            self.file.close()  # close file at end of generation

