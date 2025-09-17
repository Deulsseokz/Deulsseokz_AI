import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def classify_location(image: Image.Image, candidates: dict[str, str]):
    image_input = preprocess(image).unsqueeze(0).to(device)

    # 후보 설명만 뽑아서 텍스트 임베딩
    text_inputs = torch.cat([
        clip.tokenize(desc) for desc in candidates.values()
    ]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_match_idx = similarity.argmax().item()

    key_list = list(candidates.keys())
    desc_list = list(candidates.values())

    return {
        "location": key_list[best_match_idx],
        "description": desc_list[best_match_idx],
        "confidence": float(similarity[0][best_match_idx])
    }
