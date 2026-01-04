
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class EXAONE:
    def __init__(self):
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-4.0-1.2B",  # 1.2B로 변경
            dtype="bfloat16",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-1.2B")
    def messageSetting(self, role, content):
        self.messages = [ {"role": role, "content": content} ]
    def Start(self):
        input_ids = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        output = self.model.generate(
            input_ids.to("cuda"),
            max_new_tokens=128,
            do_sample=False,
        )
        
        return self.tokenizer.decode(output[0])


if (__name__ == "__main__"):
    ex = EXAONE()
    ex.messageSetting("user", "안녕? 너에 대해서 간단하게 소개해줘.")
    print(ex.Start())


