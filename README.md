
# Caption Recommender for Automotive Brands

This is a Streamlit app that recommends high-performing digital marketing captions for automotive brands based on historical data.

## Features
- Recommend similar captions that performed well
- Predict whether a new caption will perform well
- Filter recommendations by brand

## How to Use

### Locally
1. Place your `data_content.xlsx` into the `data/` folder
2. Run:
```bash
python prepare_data.py
python train_model.py
streamlit run streamlit_app.py
```

### Streamlit Cloud
1. Upload all files to a public GitHub repo
2. Set `streamlit_app.py` as the entry point on Streamlit Cloud
3. Add your own `data_content.xlsx` into the `data/` folder manually via GitHub or upload option

## Folder Structure
```
caption-recommender/
│
├── data/
│   └── data_content.xlsx     # <- Your input data here
│
├── models/
│   └── *.joblib              # Saved model files
│
├── output/
│   └── top_caption.csv       # Generated top captions
│
├── preprocessing.py
├── recommender.py
├── prepare_data.py
├── train_model.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```
