import os
import torch
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv() 
openai_api_key = os.getenv('OPENAI_API_KEY')

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def caption_generation(image, filename):
    model = ChatOpenAI(model="gpt-4o-mini", max_tokens=75)
    prompt = ChatPromptTemplate.from_template("""
Generate a concise, descriptive caption highlighting key objects, colors, and scenes in the image. Ensure information from the filename is included at the beginning of the caption i.e. the room name. Focus on factual content, avoiding subjective or excessive adjectives. Ensure the description reflects details in the filename if relevant to the image. 

                                              
Image: {image}
Filename: {filename}

</example1>                                    
Example: superior-king-room.jpg
Caption: Image of Superior King Room within the hotel featuring a large king-sized bed with white linens, a dark wooden headboard, and two bedside tables with lamps. A plush armchair is positioned near a window with sheer curtains, allowing natural light to illuminate the room. The walls are painted in a soft beige color, complementing the neutral-toned carpet.                                            
</example1>


</example2>                                    
Example: standard-twin-bedroom.jpg
Caption: Image of a Standard Twin Bedroom featuring two neatly made beds with white linens, a wooden nightstand between them, and a lamp. The room has a neutral color palette with beige walls and a large window allowing natural light.                                            
</example2>                                       
                                
    """)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    result = chain.invoke({"image": image, "filename": filename})

    return result

def caption_images(image_dir):
    captioned_images = {}
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith((".png", ".jpg", ".jpeg", ".JPG")):
            image_path = os.path.join(image_dir, filename)

            image = Image.open(image_path)
            image_description = "An image file named " + filename
            auto_caption = caption_generation(image, image_description)
            captioned_images[filename] = auto_caption

    return captioned_images


def find_nearest_caption(query, captions, model=model):
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)
    
    # Normalize query embedding
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    
    # Encode all captions
    caption_embeddings = model.encode(captions, convert_to_tensor=True, device=device,
                                      show_progress_bar=True, batch_size=32)
    
    # Normalize caption embeddings
    caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarities
    cosine_similarities = torch.matmul(query_embedding, caption_embeddings.T).squeeze()
    
    # Find the index of the highest similarity score
    best_match_idx = torch.argmax(cosine_similarities).item()
    
    # Return the best matching caption
    return captions[best_match_idx], cosine_similarities[best_match_idx].item()

def get_image_path_from_caption(best_caption, captioned_images, image_directory):
    # Find the filename corresponding to the best matching caption
    for filename, caption in captioned_images.items():
        if best_caption in caption:
            # Construct the full image path
            image_path = os.path.join(image_directory, filename)
            return image_path
    
    # If no matching caption is found
    return None


