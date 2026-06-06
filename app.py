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

st.markdown(
    """
    <style>
        :root {
            --bg-main: #f4efe6;
            --surface: rgba(255, 250, 242, 0.92);
            --surface-strong: #fffdf8;
            --border: rgba(73, 52, 36, 0.14);
            --text-main: #1f1a17;
            --text-soft: #6a5a4e;
            --accent: #b6542f;
            --accent-deep: #7f3518;
            --accent-soft: rgba(182, 84, 47, 0.12);
            --shadow: 0 18px 45px rgba(92, 67, 48, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(216, 124, 74, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(118, 153, 104, 0.15), transparent 26%),
                linear-gradient(180deg, #f8f2e9 0%, var(--bg-main) 100%);
            color: var(--text-main);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1f1b18 0%, #2b231f 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #f7efe7;
        }

        section[data-testid="stSidebar"] .stRadio {
            margin-bottom: 0.2rem;
        }

        section[data-testid="stSidebar"] hr {
            margin: 0.45rem 0 0.75rem;
        }

        section[data-testid="stSidebar"] h2 {
            margin-bottom: 0.9rem;
        }

        section[data-testid="stSidebar"] h3 {
            margin: 0;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            margin-bottom: 0;
        }

        .hero-card,
        .panel-card,
        .result-card,
        .insight-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
        }

        .hero-card {
            padding: 2rem 2.2rem;
            margin-bottom: 1.4rem;
        }

        .hero-kicker {
            display: inline-block;
            margin-bottom: 0.8rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-deep);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2rem, 2.7vw, 3.3rem);
            line-height: 1.02;
            font-weight: 800;
            color: var(--text-main);
        }

        .hero-body {
            margin: 0.9rem 0 0;
            max-width: 900px;
            color: var(--text-soft);
            font-size: 1.02rem;
            line-height: 1.7;
        }

        .panel-card {
            padding: 1.15rem 1.2rem 1.3rem;
            margin-bottom: 1rem;
        }

        .panel-title {
            margin: 0 0 0.35rem;
            color: var(--text-main);
            font-size: 1.02rem;
            font-weight: 700;
        }

        .panel-subtitle {
            margin: 0;
            color: var(--text-soft);
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .result-card,
        .insight-card {
            padding: 1.2rem 1.2rem 1.35rem;
            height: 100%;
        }

        .section-title {
            margin: 0 0 1rem;
            font-size: 1.08rem;
            font-weight: 800;
            color: var(--text-main);
        }

        .result-row {
            margin-bottom: 1rem;
        }

        .result-label {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.45rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .result-track {
            height: 12px;
            border-radius: 999px;
            background: rgba(73, 52, 36, 0.08);
            overflow: hidden;
        }

        .result-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #d27c4a 0%, #b6542f 55%, #7f3518 100%);
        }

        .insight-label {
            margin: 0 0 0.35rem;
            font-size: 0.84rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent-deep);
        }

        .insight-box {
            margin-bottom: 0.9rem;
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: var(--surface-strong);
            border: 1px solid rgba(73, 52, 36, 0.12);
            color: var(--text-main);
            line-height: 1.65;
            word-break: break-word;
        }

        .insight-caption {
            margin: 0;
            color: var(--text-soft);
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .stTextArea textarea {
            min-height: 180px;
            border-radius: 20px;
            border: 1px solid rgba(73, 52, 36, 0.18);
            background: rgba(255, 252, 247, 0.9);
            color: var(--text-main);
            padding: 1rem 1.1rem;
            font-size: 1rem;
            line-height: 1.6;
        }

        .stButton > button {
            min-height: 3rem;
            border: none;
            border-radius: 999px;
            background: linear-gradient(135deg, #c86539 0%, #a34622 100%);
            color: #fff8f2;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: 0 14px 30px rgba(163, 70, 34, 0.25);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #b75931 0%, #8f3818 100%);
        }

        .stRadio > div {
            gap: 0.35rem;
        }

        .stRadio label {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 0.25rem 0.55rem;
            border-radius: 14px;
        }

        .stRadio label p {
            margin: 0;
            line-height: 1.35;
        }

        .example-card {
            margin: 0.2rem 0 1rem;
            padding: 1rem 1.05rem;
            border-radius: 20px;
            background: rgba(255, 251, 246, 0.72);
            border: 1px solid rgba(73, 52, 36, 0.12);
        }

        .example-title {
            margin: 0 0 0.3rem;
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .example-copy {
            margin: 0;
            color: var(--text-soft);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        div[data-testid="stAlert"] {
            border-radius: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True,
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

EXAMPLE_TWEETS = {
    "Real Tweet 1": "My band, @MirthlessKL, is trying to save Kelantan. We're helping collect donations for flood victims in Kelantan with the amazing folks at Sudi. If you’ve got food, clothes, or cash to spare—or even just some time to share the post—please help.",
    "Real Tweet 2": "With various parts of Malaysia experiencing unhealthy air quality, here are some tips on how to survive the smoke haze",
    "Real Tweet 3": "My heartfelt condolences to the families of the victims of the Batang Kali Landslide. I hope you will find comfort and solace in the knowledge that those who have departed have gone to a better place.🥲"
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


def format_class_name(class_name):
    return class_name.replace('_', ' ').title()


if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = ""

# ==========================================
# 5. UI Layout & Logic
# ==========================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">ModernBERT + LoRA</div>
        <h1 class="hero-title">Disaster Tweet Classifier</h1>
        <p class="hero-body">
            This application demonstrates <strong>Cross-Domain Generalization</strong> using <strong>ModernBERT + LoRA</strong>.
            Select the ultimate general model, or test the model's capabilities by selecting a model that was deliberately blinded to a specific disaster during training.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for controls
with st.sidebar:
    st.markdown("## Model Configuration")
    selected_option = st.radio(
        "Choose the Active Model:",
        list(MODEL_DICT.keys())
    )
    st.markdown("---")
    st.markdown("### How this works")
    st.markdown("Instead of loading 5 massive 150M parameter models, we load **one** base model and hot-swap **5 LoRA adapters** (2.2% params each) in memory. This is highly parameter-efficient")

# Main interface
st.markdown(
    """
    <div class="panel-card">
        <p class="panel-title">Input Tweet</p>
        <p class="panel-subtitle">Enter a disaster-related tweet and run classification with the selected adapter.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

example_columns = st.columns(3)
for index, (label, example_text) in enumerate(EXAMPLE_TWEETS.items()):
    with example_columns[index]:
        if st.button(label, key=f"example_{index}", use_container_width=True):
            st.session_state["tweet_input"] = example_text

tweet_input = st.text_area(
    "Enter a disaster-related tweet:",
    height=150,
    key="tweet_input",
    placeholder="e.g., The entire street is submerged! Send rescue boats immediately."
)

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
                st.markdown('<div class="result-card"><p class="section-title">Top Categories</p>', unsafe_allow_html=True)
                for i in top_indices:
                    class_name = format_class_name(CLASSES[i])
                    prob = probs[i] * 100
                    st.markdown(
                        f"""
                        <div class="result-row">
                            <div class="result-label">
                                <span>{class_name}</span>
                                <span>{prob:.1f}%</span>
                            </div>
                            <div class="result-track">
                                <div class="result-fill" style="width: {prob:.1f}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                    
            with col2:
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <p class="section-title">Behind the Scenes (De-biasing)</p>
                        <p class="insight-label">Original</p>
                        <div class="insight-box">{html.escape(tweet_input)}</div>
                        <p class="insight-label">What the model sees</p>
                        <div class="insight-box">{html.escape(decoded_text)}</div>
                        <p class="insight-caption">Notice how disaster-specific keywords are masked as 'disaster' to force the model to learn contextual semantics instead of memorizing.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )