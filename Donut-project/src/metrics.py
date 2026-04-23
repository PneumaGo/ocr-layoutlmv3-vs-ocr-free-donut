import os
import json
import torch
import re
import editdistance
from tqdm import tqdm
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

def parse_answer(text, key):
    """
    Ultimate parser.
    First cleans the text from tokenization artifacts, then extracts the value.
    """
    pattern = rf"<s_{key}>(.*?)</s_{key}>"
    match = re.search(pattern, text, re.IGNORECASE)
    
    val = ""
    if match:
        val = match.group(1).strip()
    else:
        # Fallback logic if tags are malformed
        fallback_pattern = rf"s_{key}\s*(.*?)(?:s_|/s_{key}|$)"
        fallback_match = re.search(fallback_pattern, text, re.IGNORECASE)
        if fallback_match:
            val = fallback_match.group(1).strip()
        else:
            if text.lower().startswith(f"s{key}"):
                val = text[len(f"s{key}"):].strip()
            else:
                val = text.strip()

    # List of prefixes to remove from the beginning of the string
    prefixes_to_kill = [f"s_{key}", f"s{key}", key, "s_"]
    
    changed = True
    while changed:
        changed = False
        current_lower = val.lower()
        for prefix in prefixes_to_kill:
            if current_lower.startswith(prefix):
                val = val[len(prefix):].strip()
                current_lower = val.lower()
                changed = True
        
        if current_lower.startswith('s '):
            val = val[2:].strip()
            changed = True

    return val.strip()

def normalize_text(text):
    """
    Highly aggressive normalization for SROIE.
    Ignores differences in ampersands, special characters, and spaces.
    """
    if not text:
        return ""
    
    text = str(text).lower()
    
    # Standardize conjunctions
    text = text.replace("&amp;", "&")
    text = text.replace("and", "&") 
    
    # Remove all non-alphanumeric characters
    text = re.sub(r'[^a-z0-9]', '', text)
    
    # Remove all whitespace
    text = text.replace(" ", "")
    
    return text.strip()


def run_evaluation(model_path, test_img_dir, test_ent_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Loading model and processor
    print(f"Loading model from {model_path}...")
    processor = DonutProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()

    filenames = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    metrics = {"company": 0, "date": 0, "address": 0, "total": 0, "full_match": 0}
    similarity_scores = {"company": [], "date": [], "address": [], "total": []}
    
    total_count = len(filenames)
    print(f"Starting inference on {total_count} images...")

    for filename in tqdm(filenames):
        file_id = os.path.splitext(filename)[0]
        img_path = os.path.join(test_img_dir, filename)
        ent_path = os.path.join(test_ent_dir, f"{file_id}.txt")

        # Load Ground Truth data
        gt_data = {}
        if os.path.exists(ent_path):
            with open(ent_path, 'r', encoding='utf-8') as f:
                try:
                    gt_data = json.load(f)
                except Exception:
                    # Fallback for plain text entity files
                    f.seek(0)
                    for line in f:
                        if ":" in line:
                            k, v = line.strip().split(":", 1)
                            gt_data[k.lower().strip()] = v.strip()

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # Prepare inputs for Donut
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        task_prompt = "<s_sroie>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=768,
                num_beams=1,
                return_dict_in_generate=True,
            )

        prediction_text = processor.batch_decode(outputs.sequences)[0]
        prediction_text = prediction_text.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        
        pred_data = {}
        for k in ["company", "date", "address", "total"]:
            pred_data[k] = parse_answer(prediction_text, k)
        
        is_full_match = True
        for key in ["company", "date", "address", "total"]:
            p = normalize_text(pred_data.get(key, ""))
            g = normalize_text(gt_data.get(key, ""))

            # Calculate Normalized Edit Distance
            if g:
                dist = editdistance.eval(p, g)
                score = 1 - (dist / max(len(p), len(g))) if max(len(p), len(g)) > 0 else 0
                similarity_scores[key].append(score)
            else:
                similarity_scores[key].append(0.0)

            # Calculate Accuracy
            if p == g and g != "":
                metrics[key] += 1
            else:
                is_full_match = False
        
        if is_full_match:
            metrics["full_match"] += 1

    # Final reporting
    print("\n" + "="*50)
    print("RESULTS WITH NORMALIZATION:")
    for key in ["company", "date", "address", "total"]:
        acc = (metrics[key] / total_count) * 100
        avg_sim = (sum(similarity_scores[key]) / len(similarity_scores[key])) * 100 if similarity_scores[key] else 0
        print(f" - {key.capitalize():<10}: Acc: {acc:>6.2f}% | Avg Similarity: {avg_sim:>6.2f}%")
    
    print("-" * 50)
    print(f"Full Match: {(metrics['full_match'] / total_count) * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    MODEL_PATH = "./donut_sroie_final_model"
    TEST_IMG = "/kaggle/input/datasets/maxbegal/dataset/data/test/img"
    TEST_ENT = "/kaggle/input/datasets/maxbegal/dataset/data/test/entities"
    
    if os.path.exists(MODEL_PATH):
        run_evaluation(MODEL_PATH, TEST_IMG, TEST_ENT)
    else:
        print(f"Model not found at path: {MODEL_PATH}")