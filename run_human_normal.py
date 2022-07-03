import os
from utilities import create_result_directories, transform_txt_to_tsv

# dataset = "Switchboard-Corpus"
dataset = "Maptask-Corpus"

# print(model_res_dict)
models_list = ["human"]
human_res_dict = create_result_directories(models_list, to_local_directory=True, dataset=dataset)

if dataset == "Switchboard-Corpus":
  test_dir = './external/Switchboard-Corpus/swda_data/test/'
  speakers=["A", "B"]
else:
  test_dir = './external/Maptask-Corpus/maptask_data/test/'
  speakers=["f", "g"]

output_dir = human_res_dict["base_human"]["directory_tsv"]

if(len(os.listdir(output_dir)) > 1):
  print("Output directory:", output_dir)
  print("Output directory is not empty. Clear the directory before running the experiment.")
else:
  transform_txt_to_tsv(test_dir, "human", output_dir, speakers)