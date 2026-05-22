# 🌊 Sea Creature Classifier

An image classification web app powered by a custom **EfficientNet** model trained to identify **23 ocean species**.

## 🐠 Supported Classes

Clams · Corals · Crabs · Dolphin · Eel · Fish · Jelly Fish · Lobster · Nudibranchs · Octopus · Otter · Penguin · Puffers · Sea Rays · Sea Urchins · Seahorse · Seal · Sharks · Shrimp · Squid · Starfish · Turtle_Tortoise · Whale

## 🗂️ Project Structure

```
├── app.py                        # Streamlit app
├── efficientnet_sea_model.h5     # Trained model (add this file)
├── requirements.txt              # Python dependencies
└── README.md
```

## 🚀 Deploy on Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add app.py requirements.txt README.md
   # Add the model file (see note below)
   git add efficientnet_sea_model.h5
   git commit -m "Initial commit"
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

   > ⚠️ **Model file size**: The `.h5` file is ~17 MB which is fine for GitHub (limit is 100 MB). If it ever exceeds 100 MB, use [Git LFS](https://git-lfs.com/).

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repository, branch (`main`), and set **Main file path** to `app.py`
   - Click **Deploy**

3. **Your app will be live** at `https://<your-app-name>.streamlit.app` 🎉

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | EfficientNet (custom) |
| Input size | 224 × 224 × 3 |
| Output classes | 23 |
| Loss function | Categorical Crossentropy |
| Optimizer | Adam (lr=0.001) |
| Framework | TensorFlow / Keras 3.13 |
