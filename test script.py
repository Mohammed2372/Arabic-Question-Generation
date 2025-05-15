# libraries
import re
import torch
import sys
import io
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# input
context = 'ولد العالم العربي الشهير ابن الهيثم في مدينة البصرة عام 965 ميلادي. كان أحد أبرز العلماء في العصر الذهبي الإسلامي، حيث قدم مساهمات كبيرة في مجالات البصريات، والرياضيات، والهندسة، والفيزياء. يُعتبر كتابه "المناظر" من أهم الكتب التي أثرت في علم البصريات، حيث شرح فيه مبادئ الانعكاس والانكسار، ووضع أسس علم الضوء الحديث. بالإضافة إلى ذلك، كان لابن الهيثم دور مهم في تطوير المنهج العلمي التجريبي، إذ اعتمد على الملاحظة والتجربة لإثبات نظرياته. سافر إلى مصر بدعوة من الخليفة الفاطمي الحاكم بأمر الله، وهناك قام بدراسة سلوك الضوء في الماء والهواء.'


# functions
## remove extra spaces
def remove_extra_spaces(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    
## fix space problem in words
def fix_arabic_spacing(text):
    ### reconnect ا to the previous word if it is fully isolated
    text = re.sub(r'(\S)\s+ا\s+', r'\1ا ', text)
    ### reconnect أ and ي to the next word if they are fully isolated
    text = re.sub(r'\s+أ\s+(\S)', r' أ\1', text)
    text = re.sub(r'\s+ي\s+(\S)', r' ي\1', text)
    ### remove extra spaces
    text = remove_extra_spaces(text)
    
    return text

## remove punk
def remove_punk(text):
    arabic_punctuation = r'[،؛؟…!"#$%&\'()*+,-./:;<=>@^_`{|}~]'
    text = re.sub(arabic_punctuation, '', text)
    
    return text
## remove diacritics
def remove_diacritics(text):
    # remove diacritics from the text as it may confuse the model
    return re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    
## remove Alef variations
def remove_alef_variations(text):
    text = re.sub(r'[إأٱآ]', 'ا', text)
    return text

## apply process functions on context
def process_context(text):
    text = remove_diacritics(text)
    text = fix_arabic_spacing(text)
    text = remove_punk(text)
    text = remove_alef_variations(text)
    text = remove_extra_spaces(text)
    return text

# apply process function on context
try:
    print(context)
except UnicodeEncodeError:
    print(context.encode('utf-8').decode('utf-8'))
context_cleaned = process_context(context)
print(context_cleaned)

# load tokenizer and model
try:
    tokenizer = T5Tokenizer.from_pretrained(r"D:\GP\Question Generation\AraT5 Model\AraT5Tokenizer")
    model = T5ForConditionalGeneration.from_pretrained(r"D:\GP\Question Generation\AraT5 Model\AraT5 base final")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    sys.exit(1)

## resize model to tokenizer
model.resize_token_embeddings(len(tokenizer))
print("New model vocabulary size:", model.config.vocab_size)

## move model to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded successfully on", device)

# predict
## Initialize empty list to store questions
question_list = []

## tokenizer context
inputs = tokenizer(context_cleaned, return_tensors="pt").to(device)

## predict
for i in range(4):
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            do_sample=True,
            num_beams=1,
            early_stopping=True,
            top_p=0.8,  # Lower top_p for more diversity
            temperature=1.0,  # Higher temperature for more randomness
            num_return_sequences=1,
        )
    
    # Decode the output ids to text
    for j, output_id in enumerate(output_ids):
        generated_text = tokenizer.decode(output_id, skip_special_tokens=True)
        # Add Arabic question mark if it's not already there
        if not generated_text.endswith('؟'):
            generated_text += '؟'
        question_list.append(generated_text)
        print(f"Generated Text {i+1}-{j+1}:", generated_text)

# Print all generated questions from the list
print("\nAll generated questions:")
for idx, question in enumerate(question_list, 1):
    print(f"{idx}. {question}")