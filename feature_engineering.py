from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate_dataset_to_eng():
    df = pd.read_csv("data/train.csv")

    english_premise = df[df['lang_abv'] == 'en']['premise'].to_list()
    english_hypothesis = df[df['lang_abv'] == 'en']['hypothesis'].to_list()
    english_label= df[df['lang_abv'] == 'en']['label'].to_list()

    #Generate a fully english dataset

    for lang in list(df['lang_abv'].unique()):
        if lang != 'en':
            premise = df[df['lang_abv'] == lang]['premise'].to_list()
            hypothesis = df[df['lang_abv'] == lang]['hypothesis'].to_list()
            label= df[df['lang_abv'] == lang]['label'].to_list()

            if lang == "sw":
                model_lang = "swc"
            elif lang == "el":
                model_lang = "grk"
            else:
                model_lang = lang

            model_type = f"Helsinki-NLP/opus-mt-{model_lang}-en"
            # Initialize the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_type)

            # Initialize the model
            model = AutoModelForSeq2SeqLM.from_pretrained(model_type)

            model.cuda()

            # Tokenize text
            batch_size = 32
            
            print(lang)
            print(len(premise))

            for i in range(0, len(premise), batch_size):

                premise_batch = premise[i:i+batch_size]
                hypothesis_batch = hypothesis[i:i+batch_size]

                tokenized_premise = tokenizer(premise_batch, return_tensors='pt', padding=True).to(device)
                tokenized_hypothesis = tokenizer(hypothesis_batch, return_tensors='pt', padding=True).to(device)

                # Generate translation
                translation_premise = model.generate(**tokenized_premise)
                translation_hypothesis = model.generate(**tokenized_hypothesis)

                # print(tokenizer.batch_decode(translation_hypothesis, skip_special_tokens=True)[0])
                # print(tokenizer.batch_decode(translation_premise, skip_special_tokens=True)[0])

                # Decode model output
                english_premise += tokenizer.batch_decode(translation_premise, skip_special_tokens=True)
                english_hypothesis += tokenizer.batch_decode(translation_hypothesis, skip_special_tokens=True)
                english_label += label

    english_df = pd.DataFrame.from_dict({'english_premise' : english_premise,
                            'english_hypothesis' : english_hypothesis,
                            'english_label' : english_label})

    english_df.to_csv("data/english_df.csv")
    return english_df

def translate_to_other_languages():
    english_df = pd.read_csv("data/english_df.csv")

    english_premise = english_df['english_premise'].to_list()
    english_hypothesis = english_df['english_hypothesis'].to_list()
    english_label= english_df['english_label'].to_list()

    #Generate iterate through the different languages
    df = pd.read_csv("data/train.csv")

    for lang in list(df['lang_abv'].unique())[9:11]:
        multi_lingual_premise = []
        multi_lingual_hypothesis = []
        multi_lingual_label = []  
        print(lang)
        if lang != 'en':
            # There is no eng-thai model available in the transformers library 
            if lang == "th":
                continue
            elif lang == "tr":
                model_lang = "trk"
            else:
                model_lang = lang

            model_type = f"Helsinki-NLP/opus-mt-en-{model_lang}"
            # Initialize the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_type)

            # Initialize the model
            model = AutoModelForSeq2SeqLM.from_pretrained(model_type)

            model.cuda()

            # Tokenize text
            batch_size = 32
            
            for i in tqdm(range(0, len(english_premise), batch_size)):

                premise_batch = english_premise[i:i+batch_size]
                hypothesis_batch = english_hypothesis[i:i+batch_size]

                tokenized_premise = tokenizer(premise_batch, return_tensors='pt', padding=True).to(device)
                tokenized_hypothesis = tokenizer(hypothesis_batch, return_tensors='pt', padding=True).to(device)

                # Generate translation
                translation_premise = model.generate(**tokenized_premise)
                translation_hypothesis = model.generate(**tokenized_hypothesis)

                # Decode model output
                multi_lingual_premise += tokenizer.batch_decode(translation_premise, skip_special_tokens=True)
                multi_lingual_hypothesis += tokenizer.batch_decode(translation_hypothesis, skip_special_tokens=True)

                multi_lingual_label += english_label

                assert len(multi_lingual_premise) == len(multi_lingual_hypothesis) == len(multi_lingual_label)

        multi_lingual_df = pd.DataFrame.from_dict({'premise' : multi_lingual_premise,
                                'hypothesis' : multi_lingual_hypothesis,
                                'label' : multi_lingual_label})

        print(multi_lingual_df.shape)
        
        multi_lingual_df.to_csv(f"data/{lang}_df.csv")

    return True

if __name__ == '__main__':
    # consolidate dataset
    english_df = translate_dataset_to_eng
    english_df.columns = ['row.number', 'premise', 'hypothesis', 'label']
    df = pd.read_csv("data/train.csv")
    thai_df = df[df['lang_abv'] == "th"][['premise', 'hypothesis', 'label']]

    list_of_df = [pd.read_csv(f'data/{lang}_df.csv') for lang in list(df['lang_abv'].unique()) if lang != "en" and lang != "th"] + [thai_df] + [english_df]
    full_df = pd.concat(list_of_df)[['premise', 'hypothesis', 'label']] 
    full_df.to_csv("data/train_translated.csv")