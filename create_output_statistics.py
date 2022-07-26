from utilities import create_result_directories, get_responses_list_from_contexed_dir, create_postags_df, create_word_freq_df, create_word_concrete_df, get_dataset_settings
import spacy
import pandas as pd

print("Loading SpaCy en_core_web_sm...")
sp = spacy.load('en_core_web_sm')
print("================================ . ================================")

dataset_list = ["Switchboard-Corpus", "Maptask-Corpus"]
models_list = ["human", "DialoGPT"]
model_types_list=["base", "finetuned"]

do_pos_tags = True
do_word_freq = False
do_word_conc = False

for dataset in dataset_list:
  print("Processing Dataset:", dataset)
  test_dir, speakers = get_dataset_settings(dataset)

  for model_name in models_list:
    for model_type in model_types_list:
      model_res_dict = create_result_directories(models_list, to_local_directory=True, dataset=dataset, model_type=model_type, speakers=speakers)
      full_model_name = model_type + "_" + model_name
      if full_model_name in model_res_dict: 
        print("Processing Model:", full_model_name)

        directory_c_csv = model_res_dict[full_model_name]["directory_c_csv"]
        stats_directory = model_res_dict[full_model_name]["model_stats_directory"]

        print("0.0 Generating responses list..")
        base_responses_list = get_responses_list_from_contexed_dir(data_dir=directory_c_csv)
        base_responses_str = " ".join(base_responses_list)
        
        print("0.1 Creating responses SpaCy doc..")
        doc = sp(base_responses_str)

        # Create pos tags csvs.
        if do_pos_tags:
          print("1. Creating pos tags..")
          postags_df = create_postags_df(doc)
          postags_df["Percentage"] = postags_df.Count / postags_df.Count.sum()
          output_name = "responses_postags.csv"
          postags_df.to_csv(stats_directory + output_name)

        # Create words freq csvs.
        # all tokens that arent stop words or punctuations
        if do_word_freq:
          print("2.1 Creating word freqs of content words..")
          content_words = [token.text for token in doc
                          if not token.is_stop and not token.is_punct]
          content_words_freq_df = create_word_freq_df(content_words)
          output_name = "responses_content_words_freq.csv"
          content_words_freq_df.to_csv(stats_directory + output_name)

          print("2.2 Creating word freqs of stop words..")
          stop_words = [token.text for token in doc if token.is_stop]
          stop_words_freq_df = create_word_freq_df(stop_words)
          output_name = "responses_stop_words_freq.csv"
          stop_words_freq_df.to_csv(stats_directory + output_name)
          
          print("2.3 Creating word freqs of all words..")
          all_words = [token.text for token in doc]
          all_words_freq_df = create_word_freq_df(all_words)
          output_name = "responses_all_words_freq.csv"
          all_words_freq_df.to_csv(stats_directory + output_name)

          # external\Concreteness_ratings_Brysbaert_et_al_BRM.xlsx
          if do_word_conc:
            print("3 Creating word concretness..")
            print("3.0 Reading external csv..")
            xls = pd.ExcelFile('external\Concreteness_ratings_Brysbaert_et_al_BRM.xlsx')
            df_concreteness = xls.parse(xls.sheet_names[0])
            
            all_words = [token.text for token in doc]
            all_words_freq_df = create_word_freq_df(all_words)
            word_conrete_df = create_word_concrete_df(all_words_freq_df, df_concreteness)
            
            # Words not found were assigned a value of 0.
            word_concrete_filtered_df = word_conrete_df.loc[word_conrete_df['Conc.M'] > 0]

            output_name = "responses_word_concretness.csv"
            print("3.1 Generating output file..")
            word_concrete_filtered_df.to_csv(stats_directory + output_name)
            print("---------------- . ----------------")





