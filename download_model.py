from sentence_transformers import SentenceTransformer

# Define the model we are going to use
model_name = 'all-MiniLM-L6-v2'
# Define the local path to save it to
save_path = './models/all-MiniLM-L6-v2'

print(f"Downloading model '{model_name}'...")
model = SentenceTransformer(model_name)

print(f"Saving model to '{save_path}'...")
model.save(save_path)

print("Model saved successfully!")