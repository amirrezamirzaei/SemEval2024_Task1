import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from itertools import chain, combinations

def read_csv_dataset(path):
  raw_df = pd.read_csv(path)
  xs1, xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in raw_df.iterrows()]))
  scores = list(raw_df['Score']) if 'Score' in raw_df.columns else [None]*len(xs1)
  pairIDs = list(raw_df['PairID'])
  return [(xs1[i],xs2[i],scores[i],pairIDs[i]) for i in range(len(xs1))]

def pytorch_pearsonr(a, b):
  a = torch.tensor(a) if type(a)!=torch.Tensor else a
  b = torch.tensor(b) if type(b)!=torch.Tensor else b
  mean_a = torch.mean(a)
  mean_b = torch.mean(b)
  numerator = (a-mean_a) @ (b-mean_b)
  denumerator = torch.sqrt(torch.sum((a-mean_a)**2) * torch.sum((b-mean_b)**2))
  return numerator/denumerator

class EmbeddingDataset(Dataset):
    def __init__(self, paths):
        super(EmbeddingDataset, self).__init__()
        self.dataset = []
        for path in paths:
          with open(path, 'rb') as file:
            data_dict = pickle.load(file)
            if not self.dataset:
              self.dataset = [{}]*len(data_dict)
            i = 0
            for key,item in data_dict.items():
              if self.dataset[i]:
                self.dataset[i]['embedding'] = torch.cat((self.dataset[i]['embedding'], item[0].to(device)), dim=1)
              else:
                self.dataset[i] = {
                    'pairID'    : key,
                    'sentence1' : item[1],
                    'sentence2' : item[2],
                    'score'     : item[3],
                    'embedding' : item[0].to(device)
                }
              i+=1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence1 = self.dataset[idx]['sentence1']
        sentence2 = self.dataset[idx]['sentence2']
        score = self.dataset[idx]['score']
        embedding = self.dataset[idx]['embedding']
        pairID = self.dataset[idx]['pairID']
        return (embedding, sentence1, sentence2, score, pairID)
    

# define model
class LinearModel(nn.Module):
    def __init__(self, size):
        super(LinearModel, self).__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))


    def forward(self, x):
        x = torch.mul(x, self.weight) + self.bias
        x = F.normalize(x, p=2, dim=1)
        return x.double()

def get_scores_from_model(model, dataloader):
  model.eval()
  pred_score, target = None, None
  for batch, (embeddings, sentences1, sentences2, scores, pairsID) in enumerate(dataloader):
    with torch.no_grad():
      embeddings1 = model(embeddings[:,0,:])
      embeddings2 = model(embeddings[:,1,:])
    if pred_score is None:
      pred_score = torch.sum(torch.mul(embeddings1,embeddings2), dim=1)
      target = scores
    else:
      pred_score = torch.cat((pred_score,torch.sum(torch.mul(embeddings1,embeddings2), dim=1)))
      target = torch.cat((target,scores))
  return pred_score.cpu(), target.cpu()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

languages = ["afr", "amh", "arb", "arq", "ary", "eng", "esp", "hau", "hin", "ind", "kin", "mar", "pan", "tel"]
train_lang = 'eng' # change this to change the training language

train_dataset = EmbeddingDataset([
    f'Embeddings/original/{train_lang}_train_e5.pickle',
    f'Embeddings/original/{train_lang}_train_mbertv1.pickle',
    f'Embeddings/original/{train_lang}_train_mbertv2.pickle',
    f'Embeddings/original/{train_lang}_train_mpnetv2.pickle',
    f'Embeddings/original/{train_lang}_train_minilm.pickle',
    f'Embeddings/original/{train_lang}_train_mpnet.pickle',
   
])

dev_dataset = EmbeddingDataset([
    f'Embeddings/original/{train_lang}_dev_e5.pickle',
    f'Embeddings/original/{train_lang}_dev_mbertv1.pickle',
    f'Embeddings/original/{train_lang}_dev_mbertv2.pickle',
    f'Embeddings/original/{train_lang}_dev_mpnetv2.pickle',
    f'Embeddings/original/{train_lang}_dev_minilm.pickle',
    f'Embeddings/original/{train_lang}_dev_mpnet.pickle',                            
])

test_dataset = EmbeddingDataset([
    f'Embeddings/original/{train_lang}_test_e5.pickle',
    f'Embeddings/original/{train_lang}_test_mbertv1.pickle',
    f'Embeddings/original/{train_lang}_test_mbertv2.pickle',
    f'Embeddings/original/{train_lang}_test_mpnetv2.pickle',
    f'Embeddings/original/{train_lang}_test_minilm.pickle',
    f'Embeddings/original/{train_lang}_test_mpnet.pickle',                            
])

vector_dim = train_dataset[0][0].shape[1]
linear_model = LinearModel(vector_dim).double().to(device)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# training loop
criterion = nn.MSELoss()
learning_rate=9e-3
optimizer = torch.optim.Adam(linear_model.parameters(), lr=learning_rate)

num_epochs = 50
best_val_corr, best_epoch, best_val_test_corr = 0, 0, 0
best_val_weight, best_val_bias = None, None
first_epoch_weight, first_epoch_bias = None, None
for epoch in range(num_epochs):
    train_loss, val_loss, test_loss = 0, 0, 0
    train_corr, val_corr, test_corr = 0, 0, 0

    linear_model.train()
    for batch, (embeddings, sentences1, sentences2, scores, pairsID) in enumerate(train_loader):
        embeddings1 = linear_model(embeddings[:,0,:])
        embeddings2 = linear_model(embeddings[:,1,:])
        pred_score = torch.sum(torch.mul(embeddings1,embeddings2), dim=1)
        loss = criterion(pred_score, scores.to(device))
        train_loss += loss.item()*train_loader.batch_size
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    linear_model.eval()
    for batch, (embeddings, sentences1, sentences2, scores, pairsID) in enumerate(dev_loader):
        with torch.no_grad():
          embeddings1 = linear_model(embeddings[:,0,:])
          embeddings2 = linear_model(embeddings[:,1,:])
        pred_score = torch.sum(torch.mul(embeddings1,embeddings2), dim=1)
        loss = criterion(pred_score, scores.to(device))
        val_loss += loss.item()*dev_loader.batch_size

    pred, target = get_scores_from_model(linear_model, train_loader)
    train_corr = pytorch_pearsonr(pred, target)

    pred, target = get_scores_from_model(linear_model, dev_loader)
    val_corr = pytorch_pearsonr(pred, target)

    pred, target = get_scores_from_model(linear_model, test_loader)
    test_corr = pytorch_pearsonr(pred, target)
    if epoch==0:
      first_epoch_weight = linear_model.weight.clone()
      first_epoch_bias = linear_model.bias.clone()
    if val_corr > best_val_corr:
      best_val_corr = val_corr
      best_epoch = epoch
      best_weight = linear_model.weight.clone()
      best_bias = linear_model.bias.clone()
      best_val_test_corr = test_corr
      print(f'best val corr at epoch {best_epoch} = {best_val_corr} with test corr={test_corr}')
    train_loss/=len(train_loader)
    val_loss/=len(dev_loader)
    print(f"[Epoch {epoch}]\t"
        f"Train Loss:{train_loss:.4f}\t"
        f"Train corr:{train_corr:.4f}\t"
        f"Val Loss:{val_loss:.4f}\t"
        f"Val corr:{val_corr:.4f}\t"
        f"Test corr:{test_corr:.4f}\t")
print(f'best val corr at epoch {best_epoch} = {best_val_corr} with test corr={best_val_test_corr}')

linear_model.weight.data = first_epoch_weight
linear_model.bias.data = first_epoch_bias
linear_model.eval()

############# evaluation #############
def get_csv_from_pred(preds, path_csv):
  data = {'PairID': [],
        'Pred_Score': []}
  for PairID, Pred_Score in preds:
    data['PairID'].append(PairID)
    data['Pred_Score'].append(Pred_Score)
  df = pd.DataFrame(data)
  df[['PairID', 'Pred_Score']].to_csv(path_csv, index=False)

linear_model.eval()

print('*'*20,'original lang','*'*20)
for lang in languages:
  for type_data in ['train', 'test', 'dev']:
    try:
      dataset = EmbeddingDataset([f'Embeddings/original/{lang}_{type_data}_e5.pickle',
                                      f'Embeddings/original/{lang}_{type_data}_mbertv1.pickle',
                                      f'Embeddings/original/{lang}_{type_data}_mbertv2.pickle',
                                      f'Embeddings/original/{lang}_{type_data}_mpnetv2.pickle',
                                      f'Embeddings/original/{lang}_{type_data}_minilm.pickle',
                                      f'Embeddings/original/{lang}_{type_data}_mpnet.pickle',       
                                      ])
    except FileNotFoundError:
       continue

    preds = []
    pearson = []
    for (embeddings, sentences1, sentences2, scores, pairsID) in dataset:
      embeddings = embeddings.reshape((1,2,-1))
      with torch.no_grad():
        embeddings1 = linear_model(embeddings[:,0,:])
        embeddings2 = linear_model(embeddings[:,1,:])
      pred_score = torch.sum(torch.mul(embeddings1,embeddings2), dim=1).item()
      pearson.append((scores, pred_score))
      preds.append((pairsID,pred_score))

    get_csv_from_pred(preds, f'Result/original/pred_{lang}_a_{type_data}.csv')
    print(f'corr {lang} on {type_data}=={pytorch_pearsonr([i[0] for i in pearson], [i[1] for i in pearson])}')

print('*'*20,'translated lang','*'*20)
for lang in languages:
  for type_data in ['train', 'test', 'dev']:
    try:
      dataset = EmbeddingDataset([f'Embeddings/translations/{lang}2eng_{type_data}_e5.pickle',
                                      f'Embeddings/translations/{lang}2eng_{type_data}_mbertv1.pickle',
                                      f'Embeddings/translations/{lang}2eng_{type_data}_mbertv2.pickle',
                                      f'Embeddings/translations/{lang}2eng_{type_data}_mpnetv2.pickle',
                                      f'Embeddings/translations/{lang}2eng_{type_data}_minilm.pickle',
                                      f'Embeddings/translations/{lang}2eng_{type_data}_mpnet.pickle',       
                                      ])
    except FileNotFoundError:
       continue

    preds = []
    pearson = []
    for (embeddings, sentences1, sentences2, scores, pairsID) in dataset:
      embeddings = embeddings.reshape((1,2,-1))
      with torch.no_grad():
        embeddings1 = linear_model(embeddings[:,0,:])
        embeddings2 = linear_model(embeddings[:,1,:])
      pred_score = torch.sum(torch.mul(embeddings1,embeddings2), dim=1).item()
      pearson.append((scores, pred_score))
      preds.append((pairsID,pred_score))

    get_csv_from_pred(preds, f'Result/translations/pred_{lang}_a_{type_data}.csv')
    if pearson[0][0] is not None:
      print(f'corr {lang} on {type_data}=={pytorch_pearsonr([i[0] for i in pearson], [i[1] for i in pearson])}')