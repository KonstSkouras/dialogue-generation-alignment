from utilities import create_result_directories, create_normalized_tsv, get_dataset_settings

dataset_list = ["Switchboard-Corpus", "Maptask-Corpus"]
models_list = ["human", "DialoGPT"]
model_types_list=["base", "finetuned"]

for dataset in dataset_list:
  print("----------------------------------------------------------------")
  print("Processing Dataset:", dataset)
  test_dir, speakers = get_dataset_settings(dataset)

  for model_name in models_list:
    for model_type in model_types_list:
      model_res_dict = create_result_directories(models_list, to_local_directory=True, dataset=dataset, model_type=model_type, speakers=speakers)
      full_model_name = model_type + "_" + model_name
      if full_model_name in model_res_dict: 
        print("Processing Model:", full_model_name)

        if model_name == "human":
          # normal
          directory_tsv = model_res_dict[full_model_name]["directory_tsv"]
          directory_tsv_normalized = model_res_dict[full_model_name]["directory_tsv_normalized"]
          create_normalized_tsv(directory_tsv, directory_tsv_normalized)

          # # scrambled
          directory_tsv = model_res_dict[full_model_name]["directory_scrambled_tsv"]
          directory_tsv_normalized = model_res_dict[full_model_name]["directory_scrambled_tsv_normalized"]
          create_normalized_tsv(directory_tsv, directory_tsv_normalized)
        else:
          for speaker in speakers:
            # normal
            directory_tsv_name = "directory_tsv_" + speaker + "_M"
            directory_tsv_normalized_name = "directory_tsv_normalized_" + speaker + "_M"

            directory_tsv = model_res_dict[full_model_name][directory_tsv_name]
            directory_tsv_normalized = model_res_dict[full_model_name][directory_tsv_normalized_name]
            create_normalized_tsv(directory_tsv, directory_tsv_normalized)

            # scrambled
            if model_type == "base":
              directory_tsv_name = "directory_scrambled_tsv_" + speaker + "_M"
              directory_tsv_normalized_name = "directory_scrambled_tsv_normalized_" + speaker + "_M"

              directory_tsv = model_res_dict[full_model_name][directory_tsv_name]
              directory_tsv_normalized = model_res_dict[full_model_name][directory_tsv_normalized_name]
              create_normalized_tsv(directory_tsv, directory_tsv_normalized)

           