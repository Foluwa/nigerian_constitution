import torch
import torch.nn as nn


# load model
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


input_size = 28 # len(all_words_array) 
output_size = 5 # len(tags) 
hidden_size = 16

my_model=my_network(input_size, output_size, hidden_size)
PATH = "Nigeria.pth"
my_model.load_state_dict(torch.load(PATH))
my_model.eval()


# # get imput from user
# def user_input(message):
#     print(message)

#predict

def get_prediction(message):
        prediction=my_model(my_vector)
        prediction_probabilities=torch.softmax(prediction,dim=1)
        print('TORCH UTILS PREDIC ', prediction_probabilities)
        return prediction_probabilities
        """
        predicted_tag_index=prediction.argmax(dim=1).item()
        actual_tag_predicted=tags[predicted_tag_index]
        print(actual_tag_predicted)
        prob=prediction_probabilities[0,predicted_tag_index]
        print(prob)
        if prob<0.7:
            print(f"_bot: Sorry, I cannot understand you..." )
            continue
        #now time for chatbot to give answer
        for intent in intents["intents"]:
            if intent["tag"]==actual_tag_predicted:
                print(f"_bot:" + str(random.choice(intent["responses"])))        
        """
