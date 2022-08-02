from utilities import create_dialign_directories, create_filtered_lexicons_csvs, get_dataset_settings

dataset_list = ["Switchboard-Corpus", "Maptask-Corpus"]
models_list = ["human", "DialoGPT"]
model_types_list=["base","finetuned"]

for dataset in dataset_list:
  print("----------------------------------------------------------------")
  print("Processing Dataset:", dataset)
  test_dir, speakers = get_dataset_settings(dataset)

  for model_name in models_list:
    for model_type in model_types_list:
      dialign_normalized_dirs = create_dialign_directories(models_list, output_dir_name="dialign_normalized", to_local_directory=True, dataset=dataset, model_type=model_type, speakers=speakers)
      dialign_filtered_dirs = create_dialign_directories(models_list, output_dir_name="dialign_filtered", to_local_directory=True, dataset=dataset, model_type=model_type, speakers=speakers)
      
      full_model_name = model_type + "_" + model_name
      if full_model_name in dialign_normalized_dirs: 
        print("Processing Model:", full_model_name)

        if model_name == "human":
          # normal
          directory_input = dialign_normalized_dirs[full_model_name]["directory_normal"]
          directory_filtered_output = dialign_filtered_dirs[full_model_name]["directory_normal"]
          create_filtered_lexicons_csvs(directory_input, directory_filtered_output, filter_speaker=False, span = 6)

          # scrambled
          directory_input = dialign_normalized_dirs[full_model_name]["directory_scrambled"]
          directory_filtered_output = dialign_filtered_dirs[full_model_name]["directory_scrambled"]
          create_filtered_lexicons_csvs(directory_input, directory_filtered_output, filter_speaker=False, span = 6)
        else:
          for speaker in speakers:
            # normal
            directory_name = "directory_normal_" + speaker + "_M"

            directory_input = dialign_normalized_dirs[full_model_name][directory_name]
            directory_filtered_output = dialign_filtered_dirs[full_model_name][directory_name]
            create_filtered_lexicons_csvs(directory_input, directory_filtered_output, filter_speaker=True, span = 6)

            # scrambled
            if model_type == "base":
              directory_name = "directory_scrambled_" + speaker + "_M"

              directory_input = dialign_normalized_dirs[full_model_name][directory_name]
              directory_filtered_output = dialign_filtered_dirs[full_model_name][directory_name]
              create_filtered_lexicons_csvs(directory_input, directory_filtered_output, filter_speaker=True, span = 6)