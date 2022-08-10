from utilities import create_result_directories, create_dialign_directories, get_dataset_settings

# models_list = ["DialoGPT"]
# create_result_directories(models_list, to_local_directory=True, dataset="Switchboard-Corpus", model_type="base", speakers=["A", "B"])
# create_result_directories(models_list, to_local_directory=True, dataset="Switchboard-Corpus", model_type="finetuned", speakers=["A", "B"])
# create_result_directories(models_list, to_local_directory=True, dataset="Maptask-Corpus", model_type="base", speakers=["f", "g"])
# create_result_directories(models_list, to_local_directory=True, dataset="Maptask-Corpus", model_type="finetuned", speakers=["f", "g"])
# # print(model_res_dict)

# models_list = ["human"]
# create_result_directories(models_list, to_local_directory=True, dataset="Switchboard-Corpus")
# create_result_directories(models_list, to_local_directory=True, dataset="Maptask-Corpus", model_type="base", speakers=["f", "g"])
# # print(human_res_dict)

dataset_list = ["Switchboard-Corpus", "Maptask-Corpus"]
models_list = ["human", "DialoGPT"]
model_types_list=["base", "finetuned"]

for dataset in dataset_list:
  print("Processing Dataset:", dataset)
  test_dir, speakers = get_dataset_settings(dataset)

  for model_name in models_list:
    for model_type in model_types_list:
      create_result_directories(models_list, to_local_directory=True, dataset=dataset, model_type=model_type, speakers=speakers)
      create_dialign_directories(models_list, output_dir_name="dialign_normalized", to_local_directory=True, dataset=dataset, model_type=model_type, speakers=speakers)
      # print(dialign_dirs)