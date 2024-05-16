from transformers import BertModel
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd

class BertWithFCBefore(nn.Module):
    def __init__(self, num_labels=2):
        super(BertWithFCBefore, self).__init__()
        self.fc_before_bert = nn.Linear(5, 768)  # 添加的全连接层，输入维度为5，输出维度为768
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, input_vectors, attention_mask=None, token_type_ids=None, labels=None):
        # 通过全连接层
        fc_output = self.activation(self.fc_before_bert(input_vectors))
        # 添加一个维度以匹配BERT的输入要求
        fc_output = fc_output.unsqueeze(1)  # 使其形状从 (batch_size, hidden_size) 变为 (batch_size, sequence_length, hidden_size)
        # 通过BERT模型
        outputs = self.bert(inputs_embeds=fc_output, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return {'loss': loss, 'logits': logits}

# 加载数据
data = {
    'input': [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20]],
    'label': [1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 分割数据集
train_inputs, val_inputs, train_labels, val_labels = train_test_split(df['input'], df['label'], test_size=0.2)

train_labels = torch.tensor(train_labels.values)
val_labels = torch.tensor(val_labels.values)

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        input_vector = torch.tensor(self.inputs.iloc[idx], dtype=torch.float32)
        label = self.labels[idx]
        return {'input_vectors': input_vector, 'labels': label}

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_inputs, train_labels)
val_dataset = CustomDataset(val_inputs, val_labels)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertWithFCBefore(num_labels=2)
model.to(device)

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练函数
def train(epoch, model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:

        optimizer.zero_grad()
        input_vectors = batch['input_vectors'].to(device)
        labels = batch['labels'].to(device)
        labels = labels.unsqueeze(1)

        outputs = model(input_vectors,labels=labels)

        loss = outputs['loss']
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(loader)
    print(f'Epoch {epoch}, Training loss: {avg_loss}')

# 验证函数
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            input_vectors = batch['input_vectors'].to(device)
            labels = batch['labels'].to(device)
            labels = labels.unsqueeze(1)

            outputs = model(input_vectors,labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    print(f'Validation loss: {avg_loss}, Accuracy: {accuracy}')

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    train(epoch, model, train_loader, optimizer)
    evaluate(model, val_loader)

# 模型推理示例
sample_input = torch.tensor([[1,2,3,4,5]], dtype=torch.float32).to(device)
output = model(sample_input)
print(output)

# 保存模型
model.bert.save_pretrained('./fine_tuned_model')




























# from transformers import BertTokenizer, BertModel, BertPreTrainedModel
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from torch.optim import AdamW
# from sklearn.model_selection import train_test_split
# import pandas as pd

# class BertWithFCBefore(nn.Module):
#     def __init__(self, num_labels=2):
#         super(BertWithFCBefore, self).__init__()
#         self.fc_before_bert = nn.Linear(3, 768)  # 添加的全连接层，输入输出大小为768
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
#         self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
#         self.activation = nn.ReLU()

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
#         # 获得词嵌入
#         # inputs_embeds = self.bert.embeddings(input_ids=input_ids)
#         # 通过全连接层
#         fc_output = self.activation(self.fc_before_bert(input_ids))
#         # 通过BERT模型
#         outputs = self.bert(inputs_embeds=fc_output, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
        
#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

#         return {'loss': loss, 'logits': logits}

# # 加载数据
# data = {
#     'text': [[3,3,1],[6,6,2],[9,9,3],[12,12,4]],
#     'label': [1, 0, 1, 0]
# }
# df = pd.DataFrame(data)

# # 分割数据集
# train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)
# print(train_texts)
# print(train_labels)
# # 加载分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # 分词处理
# # train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, return_tensors='pt')
# # val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, return_tensors='pt')

# train_labels = torch.tensor(train_labels.values)
# val_labels = torch.tensor(val_labels.values)

# # 自定义数据集
# class CustomDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: val[idx] for key, val in self.encodings.items()}
#         item['labels'] = self.labels[idx]
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = CustomDataset(train_texts, train_labels)
# val_dataset = CustomDataset(val_texts, val_labels)

# # 数据加载器
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# # 设置设备
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = BertWithFCBefore(num_labels=2)
# model.to(device)
# # print(model)
# # 优化器
# optimizer = AdamW(model.parameters(), lr=5e-5)

# # 训练函数
# def train(epoch, model, loader, optimizer):
#     model.train()
#     total_loss = 0
#     for batch in loader:
#         print(batch)
#         optimizer.zero_grad()
#         input_ids = batch.values.to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs['loss']
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     avg_loss = total_loss / len(loader)
#     print(f'Epoch {epoch}, Training loss: {avg_loss}')

# # 验证函数
# def evaluate(model, loader):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for batch in loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs['loss']
#             total_loss += loss.item()
#             logits = outputs['logits']
#             predictions = torch.argmax(logits, dim=-1)
#             correct += (predictions == labels).sum().item()
#     avg_loss = total_loss / len(loader)
#     accuracy = correct / len(loader.dataset)
#     print(f'Validation loss: {avg_loss}, Accuracy: {accuracy}')

# # 训练循环
# num_epochs = 10
# for epoch in range(num_epochs):
#     train(epoch, model, train_loader, optimizer)
#     evaluate(model, val_loader)

# # 模型推理示例
# inputs = tokenizer("I hate it!", return_tensors="pt").to(device)
# output = model(**inputs)
# print(output)

# # 保存模型和分词器
# model.bert.save_pretrained('./fine_tuned_model')
# tokenizer.save_pretrained('./fine_tuned_model')
