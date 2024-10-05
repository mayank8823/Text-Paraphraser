# Paraphrasing Text Using Pegasus in Python
This repository demonstrates how to leverage Google's Pegasus model for text paraphrasing using the Hugging Face Transformers library in Python. Pegasus stands for Pre-training with Extracted Gap-sentences for Abstractive Summarization Sequence-to-sequence models, and it excels at paraphrasing text effectively.

# Installation:
To get started, make sure you have the necessary libraries installed:
pip install transformers
pip install SentencePiece

# Download Pretrained Model:

save files links are here...
tokenizer drive link:
[https://drive.google.com/drive/folders/1-Jo84xAu5d7a3opR5_pgXst2o9mFgJbq?usp=drive_link](https://drive.google.com/drive/folders/1-Jo84xAu5d7a3opR5_pgXst2o9mFgJbq?usp=sharing)
model drive link:
[https://drive.google.com/drive/folders/175r4yt92GfQ8B0RJcRTP5BQAoT734G1X?usp=drive_link](https://drive.google.com/drive/folders/175r4yt92GfQ8B0RJcRTP5BQAoT734G1X?usp=sharing)

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase')
tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')

#tokenization

text = "The ultimate test of your knowledge is your capacity to convey it to another."
batch = tokenizer([text], padding=True, truncation=True, max_length=60, return_tensors='pt')
output = model.generate(**batch, max_length=60, num_beams=5, num_return_sequences=5, temperature=1.5)
results = tokenizer.batch_decode(output, skip_special_tokens=True)
print(results)


# Predictive System

def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt")
    translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text
    
num_beams = 10
num_return_sequences = 10
context = "The ultimate test of your knowledge is your capacity to convey it to another."
results = get_response(context, num_return_sequences, num_beams)
print(results)



Feel free to explore more functionalities and adapt the code to your specific use cases!

