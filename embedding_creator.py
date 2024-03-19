from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pickle
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
import os

def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_e5_embeddings(model, tokenizer, sentences, path_save):
  embedding_dict = {}
  scores = []
  for text1,text2,score,pairID in tqdm(sentences):
    input_texts = [f'query: {text1}', f'query: {text2}']
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
      outputs = model(**batch_dict)
    embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1)
    scores.append((embedding[0,:] @ embedding[1,:]).item())
    embedding_dict[pairID] = (embedding, text1, text2, score)
  with open(path_save, 'wb') as handle:
    pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def get_sentence_transformer_embedding(model, sentences, path_save):
  embedding_dict = {}
  scores = []
  for text1,text2,score,pairID in tqdm(sentences):
    input_texts = [text1.lower(), text2.lower()]
    with torch.no_grad():
      embedding = torch.tensor(model.encode(input_texts))
    embedding = F.normalize(embedding, p=2, dim=1)
    scores.append((embedding[0,:] @ embedding[1,:]).item())
    embedding_dict[pairID] = (torch.tensor(embedding).clone().detach(), text1, text2, score)
  with open(path_save, 'wb') as handle:
    pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_csv_dataset(path):
    raw_df = pd.read_csv(path)
    if 'translations' in path:
        xs1 = list(raw_df['Text1 Translation'])
        xs2 = list(raw_df['Text2 Translation'])
    elif 'STS' in path:
        xs1 = list(raw_df['Text1'])
        xs2 = list(raw_df['Text2'])
    else:
        try:
            xs1, xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in raw_df.iterrows()]))
        except:
            xs1, xs2 = map(list, zip(*[tuple(row['Text'].split('\t')) for idx, row in raw_df.iterrows()]))
    scores = list(raw_df['Score']) if 'Score' in raw_df.columns else [None]*len(xs1)
    pairIDs = list(raw_df['PairID'])
    return [(xs1[i],xs2[i],scores[i],pairIDs[i]) for i in range(len(xs1))]
  
languages = ["afr", "amh", "arb", "arq", "ary", "eng", "esp", "hau", "hin", "ind", "kin", "mar", "pan", "tel"]


# target_lang = "tel2eng"

paths = []
paths.append((f'/home/amirzaei/code/SemEval2024_Task1/Data/STS/track4a_sts_esp_eng_test.csv', f'Embeddings/similarity/track4a_sts_esp_eng_test_MODELNAME.pickle'))
paths.append((f'/home/amirzaei/code/SemEval2024_Task1/Data/STS/track4b_sts_esp_eng_test.csv', f'Embeddings/similarity/track4b_sts_esp_eng_test_MODELNAME.pickle'))
paths.append((f'/home/amirzaei/code/SemEval2024_Task1/Data/STS/track5_eng_eng_test.csv', f'Embeddings/similarity/track5_eng_eng_test_MODELNAME.pickle'))
paths.append((f'/home/amirzaei/code/SemEval2024_Task1/Data/STS/trans_track4a_sts_esp_eng_test.csv', f'Embeddings/similarity/trans_track4a_sts_esp_eng_test_MODELNAME.pickle'))
paths.append((f'/home/amirzaei/code/SemEval2024_Task1/Data/STS/trans_track4b_sts_esp_eng_test.csv', f'Embeddings/similarity/trans_track4b_sts_esp_eng_test_MODELNAME.pickle'))
paths.append((f'/home/amirzaei/code/SemEval2024_Task1/Data/STS/trans_eng_esp_dev.csv', f'Embeddings/similarity/trans_eng_esp_dev_MODELNAME.pickle'))
paths.append((f'/home/amirzaei/code/SemEval2024_Task1/Data/STS/eng_esp_dev.csv', f'Embeddings/similarity/eng_esp_dev_MODELNAME.pickle'))
for lang in languages:
    paths.append((f'Data/original/{lang}_train.csv', f'Embeddings/original/{lang}_train_MODELNAME.pickle'))
    paths.append((f'Data/original/{lang}_test_with_labels.csv', f'Embeddings/original/{lang}_test_MODELNAME.pickle'))
    paths.append((f'Data/original/{lang}_dev_with_labels.csv', f'Embeddings/original/{lang}_dev_MODELNAME.pickle'))
    paths.append((f'Data/translations/{lang}2eng_test.csv', f'Embeddings/translations/{lang}2eng_test_MODELNAME.pickle'))
    paths.append((f'Data/translations/{lang}2eng_train.csv', f'Embeddings/translations/{lang}2eng_train_MODELNAME.pickle'))
    paths.append((f'Data/translations/{lang}2eng_dev.csv', f'Embeddings/translations/{lang}2eng_dev_MODELNAME.pickle'))

# E5 embedding
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to('cuda')
for path in paths:
   if not os.path.exists(path[0]) or os.path.exists(path[1].replace('MODELNAME', 'e5')):
     continue
   sentences = read_csv_dataset(path[0])
   get_e5_embeddings(model, tokenizer, sentences, path[1].replace('MODELNAME', 'e5'))
del model
del tokenizer
torch.cuda.empty_cache()


# sentence embedding models


model_names = [
   ('sentence-transformers/distiluse-base-multilingual-cased-v1', 'mbertv1'),
   ('sentence-transformers/distiluse-base-multilingual-cased-v2', 'mbertv2'),
   ('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'mpnetv2'),
   ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'minilm'),
   ('sentence-transformers/all-mpnet-base-v2', 'mpnet')
]
for model_name, save_name in model_names:
  model = SentenceTransformer(model_name).to('cuda')
  for path in paths:
   if not os.path.exists(path[0]) or os.path.exists(path[1].replace('MODELNAME', save_name)):
     continue
   sentences = read_csv_dataset(path[0])
   get_sentence_transformer_embedding(model , sentences, path[1].replace('MODELNAME', save_name))
  del model
  torch.cuda.empty_cache()