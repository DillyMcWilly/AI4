# Install necessary libraries (this section is only needed for MyBinder)
!pip install transformers==4.10.0
!pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1

# Import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Define the Chatbot class
class Chatbot:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.chat_history = []

    def respond(self, user_input):
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95
            )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.chat_history.append((user_input, response))
        return response

    def chat(self):
        print("AI: Hello! I'm your chatbot. Ask me anything or just start a conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("AI: Goodbye!")
                break
            response = self.respond(user_input)
            print("AI:", response)

# Create an instance of the Chatbot class and start the chat
chatbot = Chatbot()
chatbot.chat()
