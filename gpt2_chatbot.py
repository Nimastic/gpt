from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def generate_response(prompt, max_length=100, temperature=0.7, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length + input_ids.size(1),
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response[len(prompt):]
    return response.strip()

def chat():
    print("Welcome to the GPT-2 Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if not user_input:
            continue
        
        prompt = f"User: {user_input}\nAssistant:"
        response = generate_response(prompt)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    chat()
