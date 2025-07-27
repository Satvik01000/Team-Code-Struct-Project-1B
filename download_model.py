# from sentence_transformers import SentenceTransformer

# # Define the model we are going to use
# model_name = 'all-mpnet-base-v2'
# # Define the local path to save it to
# save_path = './models/all-mpnet-base-v2'

# print(f"Downloading model '{model_name}'...")
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# print(f"Saving model to '{save_path}'...")
# model.save(save_path)
# print("Model saved successfully!")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-base-v2")
model.save("./models/e5-base-v2")
