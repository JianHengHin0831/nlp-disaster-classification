import streamlit as st
import torch
import re
import html
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Disaster Response AI",
    page_icon="🌪️",
    layout="wide"
)

# ==========================================
# 2. Hardcoded Classes & Variables
# ==========================================
# Exact 10 classes from HumAID dataset in alphabetical order
CLASSES = [
    'caution_and_advice', 
    'displaced_people_and_evacuations', 
    'infrastructure_and_utility_damage', 
    'injured_or_dead_people', 
    'missing_or_found_people', 
    'not_humanitarian', 
    'other_relevant_information', 
    'requests_or_urgent_needs', 
    'rescue_volunteering_or_donation_effort', 
    'sympathy_and_support'
]

MODEL_NAME = 'answerdotai/ModernBERT-base'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map UI dropdown options to adapter names and folder paths
MODEL_DICT = {
    "General Model (Trained on 100% Data - Best for Future Disasters)": {
        "adapter_name": "general",
        "path": "saved_lora_models/lora_general_all_data"
    },
    "Experiment: Unseen Earthquake": {
        "adapter_name": "no_earthquake",
        "path": "saved_lora_models/lora_no_earthquake"
    },
    "Experiment: Unseen Fire": {
        "adapter_name": "no_fire",
        "path": "saved_lora_models/lora_no_fire"
    },
    "Experiment: Unseen Flood": {
        "adapter_name": "no_flood",
        "path": "saved_lora_models/lora_no_flood"
    },
    "Experiment: Unseen Hurricane": {
        "adapter_name": "no_hurricane",
        "path": "saved_lora_models/lora_no_hurricane"
    }
}

# ==========================================
# 3. Model Loading (Cached to load only once)
# ==========================================
@st.cache_resource(show_spinner="Loading ModernBERT Base & LoRA Adapters...")
def load_all_models():
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load Base Model (Only 1 Base Model in memory!)
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(CLASSES))
    base_model.to(DEVICE)
    
    # Load the First Adapter (General) to initialize PeftModel
    general_path = MODEL_DICT["General Model (Trained on 100% Data - Best for Future Disasters)"]["path"]
    model = PeftModel.from_pretrained(base_model, general_path, adapter_name="general")
    
    # Load the rest of the adapters into the SAME model dynamically
    for key, info in MODEL_DICT.items():
        if info["adapter_name"] != "general":
            try:
                model.load_adapter(info["path"], adapter_name=info["adapter_name"])
            except Exception as e:
                st.warning(f"Adapter {info['adapter_name']} not found at {info['path']}. Please check your folders.")
                
    model.eval()
    return tokenizer, model

tokenizer, model = load_all_models()

# ==========================================
# 4. Text Preprocessing (De-biasing)
# ==========================================
def clean_tweet(text):
    if not text: return ""
    text = html.unescape(str(text)).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+', '', text)
    
    disaster_keywords = ['earthquake', 'quake', 'fire', 'wildfire', 'flood', 'flooding', 'hurricane', 'storm', 'cyclone', 'typhoon']
    for kw in disaster_keywords:
        text = text.replace(kw, ' disaster ')
        
    text = text.replace('#', '')
    text = re.sub(r'(.)\1+', r'\1\1', text) 
    return " ".join(text.split())

# ==========================================
# 5. UI Layout & Logic
# ==========================================
st.title("Zero-Shot Disaster Tweet Classifier")
st.markdown("""
This application demonstrates **Cross-Domain Generalization** using **ModernBERT + LoRA**. 
Select the ultimate general model, or test the model's zero-shot capabilities by selecting a model that was deliberately blinded to a specific disaster during training.
""")

# Sidebar for controls
with st.sidebar:
    st.header("Model Configuration")
    selected_option = st.radio(
        "Choose the Active Model:",
        list(MODEL_DICT.keys())
    )
    st.markdown("---")
    st.markdown("**How this works:**")
    st.markdown("Instead of loading 5 massive 150M parameter models, we load **one** base model and hot-swap **5 LoRA adapters** (2.2% params each) in memory. This is highly parameter-efficient")

# Main interface
tweet_input = st.text_area("Enter a disaster-related tweet:", height=150, placeholder="e.g., The entire street is submerged! Send rescue boats immediately.")

if st.button("Analyze Tweet", use_container_width=True):
    if not tweet_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # 1. Clean & De-bias Text
            cleaned_text = clean_tweet(tweet_input)
            
            # 2. Switch to the selected LoRA adapter instantly
            active_adapter = MODEL_DICT[selected_option]["adapter_name"]
            model.set_adapter(active_adapter)
            
            # 3. Tokenize & Predict
            inputs = tokenizer(cleaned_text, return_tensors="pt", max_length=128, truncation=True).to(DEVICE)

            decoded_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
            
            # 4. Sort and get top 3 predictions
            top_indices = np.argsort(probs)[::-1][:3]
            
            # --- DISPLAY RESULTS ---
            st.markdown("### Prediction Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Top Categories:**")
                for i in top_indices:
                    class_name = CLASSES[i].replace('_', ' ').title()
                    prob = probs[i] * 100
                    st.progress(int(prob), text=f"**{class_name}** ({prob:.1f}%)")
                    
            with col2:
                # Presentation Hack: Show the reviewer how the text was masked!
                st.markdown("**Behind the Scenes (De-biasing)**")
                st.info(f"**Original:** {tweet_input}")
                st.success(f"**What the model sees:** {decoded_text}")
                st.caption("*Notice how disaster-specific keywords are masked as 'disaster' to force the model to learn contextual semantics instead of memorizing.*")