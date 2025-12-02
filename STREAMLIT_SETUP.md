# 🚀 Streamlit Cloud Setup Guide

## Quick Setup for Streamlit Cloud

### Step 1: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set the main file to: `app.py`
4. Click "Deploy"

### Step 2: Configure Secrets (IMPORTANT!)

Once deployed, you need to add your FMP API key:

1. Click on **Settings** (gear icon) in your deployed app
2. Go to **Secrets** section
3. Add the following content:

```toml
FMP_API_KEY = "your_actual_api_key_here"
```

4. Click **Save**
5. The app will automatically restart with the secret loaded

### Step 3: Get Your FMP API Key

1. Go to https://financialmodelingprep.com/developer/docs/
2. Sign up for a free account
3. Copy your API key
4. Paste it in the Streamlit secrets (Step 2)

---

## Local Development Setup

### Option 1: Using Secrets (Recommended)

1. Create the file `.streamlit/secrets.toml` (not tracked by git):

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

2. Edit `.streamlit/secrets.toml` and add your API key:

```toml
FMP_API_KEY = "your_api_key_here"
```

3. Run the app:

```bash
streamlit run app.py
```

### Option 2: Manual Entry

Just run the app and enter your API key in the sidebar:

```bash
streamlit run app.py
```

---

## File Structure

Make sure your repository has this structure:

```
evolution_analysis/
├── app.py                    # Main Streamlit app (use this, not streamlit_app.py)
├── main.py                   # CLI interface
├── requirements.txt          # Python dependencies
├── config/
│   └── config.yaml
├── .streamlit/
│   ├── config.toml          # Streamlit config (committed)
│   ├── secrets.toml.example # Example secrets (committed)
│   └── secrets.toml         # Your actual secrets (NOT committed)
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── fmp_client.py
    │   └── data_loader.py
    ├── indicators/
    │   ├── __init__.py
    │   ├── base.py
    │   └── technical.py
    ├── backtesting/
    │   ├── __init__.py
    │   ├── engine.py
    │   ├── position.py
    │   └── metrics.py
    ├── genetic/
    │   ├── __init__.py
    │   ├── chromosome.py
    │   ├── population.py
    │   ├── operators.py
    │   ├── fitness.py
    │   └── evolution.py
    ├── validation/
    │   ├── __init__.py
    │   └── validator.py
    └── utils/
        ├── __init__.py
        └── reporting.py
```

---

## Troubleshooting

### Error: "No module named 'src.data'"

**Solution**: Make sure all `__init__.py` files exist in each directory:
```bash
# Create missing __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/indicators/__init__.py
touch src/backtesting/__init__.py
touch src/genetic/__init__.py
touch src/validation/__init__.py
touch src/utils/__init__.py
```

### Error: "API key not found"

**Solution**: Configure secrets as described in Step 2 above.

### App is slow or times out

**Solution**: Reduce population size and generations in the sidebar:
- Population: 50 (instead of 100)
- Generations: 25 (instead of 50)

This will make it faster but less thorough.

---

## Performance Tips for Streamlit Cloud

Streamlit Cloud has resource limitations. For better performance:

1. **Start small**: Use smaller population (50) and fewer generations (25)
2. **Test with short date ranges**: Start with 2-3 years of data
3. **Skip walk-forward**: It's computationally intensive
4. **Use a single symbol**: Don't try multiple symbols at once

Once you verify it works, you can increase parameters.

---

## Example Configurations

### Fast Test (2-3 minutes)
- Population: 50
- Generations: 25
- Date range: 2022-01-01 to 2024-12-31

### Standard Run (5-10 minutes)
- Population: 100
- Generations: 50
- Date range: 2019-01-01 to 2024-12-31

### Thorough Analysis (15-20 minutes)
- Population: 200
- Generations: 75
- Date range: 2018-01-01 to 2024-12-31

---

## Need Help?

If you encounter issues:

1. Check the error message in the Streamlit app
2. Review the file structure (see above)
3. Verify your API key is correct
4. Try with smaller parameters first
5. Check Streamlit Cloud logs

---

Happy Trading! 📈
