import os
from utilities import create_result_directories, transform_txt_to_tsv, get_dataset_settings

# dataset = "Switchboard-Corpus"
dataset = "Maptask-Corpus"

# print(model_res_dict)
models_list = ["human"]
human_res_dict = create_result_directories(models_list, to_local_directory=True, dataset=dataset)

test_dir, speakers = get_dataset_settings(dataset)

output_dir = human_res_dict["base_human"]["directory_tsv"]

if(len(os.listdir(output_dir)) > 1):
  print("Output directory:", output_dir)
  print("Output directory is not empty. Clear the directory before running the experiment.")
else:
  transform_txt_to_tsv(test_dir, "human", output_dir, speakers)