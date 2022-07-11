import os
from utilities import create_result_directories, test_all, get_dataset_settings
from transformers import AutoModelWithLMHead, AutoTokenizer

# dataset = "Switchboard-Corpus"
dataset = "Maptask-Corpus"
model_type = "base"

models_list = ["DialoGPT"]

# data_dir = './external/Switchboard-Corpus/swda_data/test/'
test_dir, speakers = get_dataset_settings(dataset)
model_res_dict = create_result_directories(models_list, to_local_directory=True, dataset=dataset, model_type="base", speakers=speakers)

for model_name in models_list:
  full_model_name = model_type + "_" + model_name
  directory_c_csv = model_res_dict[full_model_name]["directory_c_csv"]

  if(len(os.listdir(directory_c_csv)) > 1):
    print("Output directory:", directory_c_csv)
    print("Output directory is not empty. Clear the directory before running the experiment.")
  else:
    model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    test_all(model, model_name, model_type, tokenizer, model_res_dict, verbose = False, data_dir = test_dir, speakers=speakers)