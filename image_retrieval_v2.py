import os
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch

# move model to device if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Load CLIP Model
clip_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_model.to(device)

#Load Sentence Transformer
st_model_id = "sentence-transformers/all-MPNet-base-v2"
st_model = SentenceTransformer(st_model_id)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st_model.to(device)

def load_images_from_folder(folder_path, target_size=(224, 224)):
    images = []
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg", ".JPG")):
            img_path = os.path.join(folder_path, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = Resize(target_size)(img)
                    img_tensor = to_tensor(img)
                    images.append(img_tensor)
                    image_paths.append(img_path)
            except IOError:
                print(f"Error loading image {filename}")
    return torch.stack(images), image_paths

def generate_image_embedding(image_batch):
    processed_images = clip_processor(
    text=None,
    images=image_batch,
    return_tensors='pt',
    do_rescale=False
)['pixel_values'].to(device)
    img_emb = clip_model.get_image_features(processed_images)
    img_emb = img_emb.detach().numpy()
    img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    return img_emb

def process_query(query_text):
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features

def retrieve_images_clip(query_text, image_features, image_paths, top_k=3):
    # Process and embed the query
    query_features = process_query(query_text)

        # Convert image_features to a PyTorch tensor if it is a NumPy array
    if isinstance(image_features, np.ndarray):
        image_features = torch.tensor(image_features)
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)
    
    # Compute similarities
    similarities = (query_features @ image_features.T).squeeze()
    
    # Get top-k matches
    top_scores, top_indices = similarities.topk(top_k)
    
    results = [
        (image_paths[i], float(score))  # Convert to float explicitly
        for i, score in zip(top_indices.tolist(), top_scores.tolist())
    ]
    
    return results


def caption_generation(image, image_path):
    model = ChatOpenAI(model="gpt-4o", max_tokens=150)
    prompt = ChatPromptTemplate.from_template("""
<CAPTION_GENERATION>
    <INSTRUCTIONS>
        Generate a concise, detailed caption that accurately describes the key objects, colors, and scene elements in the image. Focus on factual content, avoiding subjective adjectives or opinions. Max output token count is 150, do not exceed. 
    </INSTRUCTIONS>
    <STRUCTURE>
        <OBJECTS>Identify the primary and secondary objects in the image.</OBJECTS>
        <COLORS>Mention the dominant colors visible in the image.</COLORS>
        <SCENE>Describe the setting or location where the image takes place.</SCENE>
    </STRUCTURE>
    <FILENAME_INTEGRATION>
        <FILENAME>Image: {image}</FILENAME>
        <IMAGE_PATH>Filename: {image_path}</IMAGE_PATH>
    </FILENAME_INTEGRATION>
    <EXAMPLE>
        Example format for the caption:
        <EXAMPLE_TEXT> A rectangular conference table, referred to as Omega from filename, with black chairs arranged around it, each chair facing a paper and glass of water. The room features a red carpet and grey walls. A blank projector screen is mounted at the front, with a flipchart positioned to the side.
    </EXAMPLE>
</CAPTION_GENERATION>
    """)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    result = chain.invoke({"image": image, "image_path": image_path})

    return result


def caption_images(top_5_paths):
    captioned_images = {}
    for image_path in top_5_paths:
        if image_path.endswith((".png", ".jpg", ".jpeg", ".JPG")):
            image = Image.open(image_path)
            auto_caption = caption_generation(image, image_path)
            captioned_images[image_path] = auto_caption

    return captioned_images

def get_most_relevant_image(query, image_paths, img_embeddings, captioned_images):
    
    top_k_results = retrieve_images_clip(query, img_embeddings, image_paths, top_k=3)
    top_k_paths = []
    for path, _ in top_k_results:
        top_k_paths.append(path)
    captions = list(captioned_images.values())


    query_embedding = st_model.encode(query, convert_to_tensor=True, device=device)
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    caption_embeddings = st_model.encode(captions, convert_to_tensor=True, device=device, show_progress_bar=True, batch_size=32)
    caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=-1, keepdim=True)

    cosine_similarities = torch.matmul(query_embedding, caption_embeddings.T).squeeze()
    top_k = min(5, len(captions))  # Get top 5 or all if less than 5
    top_similarities, top_indices = torch.topk(cosine_similarities, k=top_k)

    # Create pairs of (index, similarity)
    index_similarity_pairs = list(zip(top_indices.tolist(), top_similarities.tolist()))

    # The best match is still the first one in the list
    best_match_idx = index_similarity_pairs[0][0]
    best_caption = captions[best_match_idx]

    return best_match_idx, best_caption



def get_image_path_from_caption(best_caption, captioned_images):
    # Find the filename corresponding to the best matching caption
    for path, caption in captioned_images.items():
        print(caption)
        if caption == best_caption:
            return path





