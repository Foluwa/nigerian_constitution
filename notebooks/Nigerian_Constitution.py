import json
import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

# Incase punkt is not found uncomment this line and rerun
nltk.download('punkt')

# Tokenizer

def tokenize(my_sentence):
    return nltk.word_tokenize(my_sentence)

# Stemmer
from nltk.stem.porter import PorterStemmer
stemmer= PorterStemmer()

def stemming(single_word):
    return stemmer.stem(single_word.lower())

def bag_of_words_converter(tokenized_stemmed_sentence,bag_of_words_original):
    
    vector=np.zeros(len(bag_of_words_original),dtype=np.float32)
    for word in tokenized_stemmed_sentence:
        if word in bag_of_words_original:
            my_index=bag_of_words_original.index(word)
            vector[my_index]=1
    return vector

# load intents
with open('../data/constitution_intents.json','r') as f:
    intents=json.load(f)

# print(intent)

text_tag_tuple=[]
tags=[]
all_words_array=[]
for intent in intents["intents"]:
    tag=intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        tokenized_sentence=tokenize(pattern)
        text_tag_tuple.append((tokenized_sentence,tag))
        all_words_array.extend(tokenized_sentence)


# now making sorted and unieqe set of all_words array
other_characters=['?','!','.', ',']
tags=sorted(set(tags))

# Stemming each words and character removal
all_words_array=[stemming(w) for w in all_words_array if w not in other_characters]
all_words_array=sorted(set(all_words_array))

    

my_sen="howsf aresafasf yousfx"
tok=tokenize(my_sen)
print(tok)
stem_version=[stemming(w) for w in tok]
print(stem_version)

import glob
x=glob.glob('../input/my-data1212/*')

X_train=[]
Y_train=[]

for (sentence,tag) in text_tag_tuple:
    sentence=[stemming(w) for w in sentence]
    my_converted_vector=bag_of_words_converter(sentence,all_words_array)
    index_of_label=tags.index(tag)
    X_train.append(my_converted_vector)
    Y_train.append(index_of_label)
X_train=np.array(X_train)
Y_train=np.array(Y_train)

# pytorch network
class chatbot_dataset(Dataset):
    def __init__(self):
        self.length_data=len(X_train)
        self.x_data=X_train
        self.y_data=Y_train
    def __getitem__(self,idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return len(X_train)

dataset_created=chatbot_dataset()
train_loader=DataLoader(dataset_created,batch_size=8,shuffle=True,num_workers=2)

#now lets make the neural network

class my_network(nn.Module):
    def __init__(self,input_size,out_classes,hidden_size):
        super().__init__()
        self.linear1=nn.Linear(in_features=input_size,out_features=hidden_size)
        self.linear2=nn.Linear(in_features=hidden_size,out_features=hidden_size)
        self.linear3=nn.Linear(in_features=hidden_size,out_features=out_classes)
        self.relu=nn.ReLU()
        
    def forward(self,t):
        t=t
        t=self.linear1(t)
        t=self.relu(t)
        t=self.linear2(t)
        t=self.relu(t)
        t=self.linear3(t)

        return t

#now defining some basic variables
input_size=len(all_words_array)
output_size=len(tags)
hidden_size=16
#defining loss, optimizer etc
#Now making arguments for above ftn


my_model=my_network(input_size,output_size,hidden_size)
learning_rate1=0.003
my_loss=nn.CrossEntropyLoss()
my_optimizer=torch.optim.Adam(my_model.parameters(),lr=learning_rate1)
dynamic_learning_rate=torch.optim.lr_scheduler.StepLR(my_optimizer,step_size=7,gamma=0.1)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#now time to make the model 
def my_train_model(model,data,optimizer,given_loss,scheduler,total_epochs=1000):

    train_loss , train_acc, val_loss, val_accuracy = [],[],[],[]

    my_sizes={ 'train': len(X_train)}
    #first loop for the epochs
    for i in range (total_epochs):
            total_correct=0
            for batch in data:
                
                
                #now performing the forward steps 
                input_data,labels=batch
                #put data into GPU processing if available
                input_data=input_data.to(device)
                labels=labels.to(device)
                my_prediction=model(input_data)
                #find loss
                loss=given_loss(my_prediction,labels)
                total_correct+=my_prediction.argmax(dim=1).eq(labels).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
#             if (i+1 % 100 == 0):
            print(f'epoch {i+1}/1000,loss={loss.item():.4f}')
            print(' Accuracy= ' +  str(total_correct/my_sizes["train"]))

    return model

trained_model=my_train_model(model=my_model.to(device),data=train_loader,
               optimizer=my_optimizer,
                    given_loss=my_loss,
               scheduler=dynamic_learning_rate)

torch.save(trained_model.state_dict(), "Nigeria.pth")
torch.save(model.state_dict(), "Nigeria.pth")


#now training complete 
#implement a chat environment
my_model.eval()
device=torch.device("cpu")

import random
print('NaijaBot is ready to chat with you !!! / enter "quit" to leave the chatting ')
print('lets start !')
while True:
    sentence = input('you :')
    if sentence == "quit":
        break
    #now tokenize ,stem and feed to network
    tokenized=tokenize(sentence)
    stemmed=[stemming(w) for w in tokenized]
    my_vector=bag_of_words_converter(stemmed,all_words_array)
    
    my_vector=torch.from_numpy(my_vector)
    my_vector=torch.unsqueeze(my_vector,0)
    my_vector.to(device)
    trained_model.to(device)
    prediction=trained_model(my_vector)
    prediction_probabilities=torch.softmax(prediction,dim=1)
    predicted_tag_index=prediction.argmax(dim=1).item()
#     print(predicted_tag_index)
    actual_tag_predicted=tags[predicted_tag_index]
#     print(actual_tag_predicted)
    #also checking for probability so that it doesnot give unwanted answers
    prob=prediction_probabilities[0,predicted_tag_index]
#     print(prob)
    if prob<0.7:
        print(f"_bot: Sorry, I cannot understand you..." )
        continue
    
    #now time for chatbot to give answer
    for intent in intents["intents"]:
        if intent["tag"]==actual_tag_predicted:
            print(f"_bot:" + str(random.choice(intent["responses"])))


# torch.save(model.state_dict(), "Nigeria.pth")