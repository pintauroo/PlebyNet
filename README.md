# PlebyNet


todo

plot against alibaba allocator. we show that we do not exceed the bottleneck links while they do





#Here we are setting the tokenizer, we setted the padding side to the right and the pad token to the eos token as padding
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForQuestionAnswering, AutoModelForTokenClassification, AutoModelForMultipleChoice, AutoModelForNextSentencePrediction, AutoModelForTableQuestionAnswering, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForMultipleChoice, AutoModelForNextSentencePrediction, AutoModelForTableQuestionAnswering, AutoModelForQuestionAnswering
from huggingface_hub import login
# token='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'
token='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'


login(token=token)
model_name='mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"