import os
from utilities import create_result_directories, get_dialogue_uttr_dict, create_scrambled_dialogues_txt, transform_txt_to_tsv, get_dataset_settings

# dataset = "Switchboard-Corpus"
dataset = "Maptask-Corpus"

# print(model_res_dict)
models_list = ["human"]
human_res_dict = create_result_directories(models_list, to_local_directory=True, dataset=dataset)

test_dir, speakers = get_dataset_settings(dataset)
# print("human_res_dict", human_res_dict)
output_dir_txt = human_res_dict["base_human"]["directory_scrambled_txt"]
output_dir_tsv = human_res_dict["base_human"]["directory_scrambled_tsv"]

all_dialogue_uttr_dict = get_dialogue_uttr_dict(test_dir)

if(len(os.listdir(output_dir_tsv)) > 1):
  print("Output directory:", output_dir_tsv)
  print("Output directory is not empty. Clear the directory before running the experiment.")
else:
  create_scrambled_dialogues_txt(all_dialogue_uttr_dict, output_dir_txt, speakers) 
  transform_txt_to_tsv(output_dir_txt, "human", output_dir_tsv)
