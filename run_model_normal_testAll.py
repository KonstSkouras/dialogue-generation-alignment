from utilities import create_result_directories, test_all
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

models_list = ["DialoGPT"]
model_res_dict = create_result_directories(models_list)

data_dir = './external/Switchboard-Corpus/swda_data/test/'

test_all(model, 'DialoGPT', tokenizer, model_res_dict, verbose = False, data_dir = data_dir)