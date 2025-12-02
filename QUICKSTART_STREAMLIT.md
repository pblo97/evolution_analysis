# ⚡ Quick Start: Streamlit Cloud Deployment

## 📋 Checklist (5 minutes)

### ✅ Step 1: Deploy to Streamlit Cloud (2 min)

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Repository: `pblo97/evolution_analysis`
   - Branch: `claude/genetic-algo-trading-strategy-01GFJ6gv8drX2VnFP7je7yCf` (or main)
   - Main file path: `app.py`
5. Click "Deploy!"

### ✅ Step 2: Configure API Key (2 min)

While the app is deploying:

1. Get your FMP API key:
   - Go to https://financialmodelingprep.com
   - Sign up (free)
   - Copy your API key

2. In Streamlit Cloud:
   - Click the **Settings** (⚙️) button
   - Go to **Secrets**
   - Paste this:
   ```toml
   FMP_API_KEY = "paste_your_key_here"
   ```
   - Click **Save**

### ✅ Step 3: Use the App! (1 min)

1. Wait for deployment to finish (~2-3 minutes)
2. Your app will open automatically
3. It will show "✅ API Key loaded from secrets"
4. Start trading strategy optimization!

---

## 🎯 First Test Run

**Recommended Settings for First Test:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Symbol | AAPL | Popular, stable stock |
| Start Date | 2022-01-01 | 3 years of data |
| End Date | 2024-12-31 | Recent data |
| Population | 50 | Fast computation |
| Generations | 25 | Quick results |

**Expected time:** 3-5 minutes

---

## 🚨 Troubleshooting

### Error: "No module named 'src'"
**Solution**: Make sure `app.py` is the main file (not `streamlit_app.py`)

### Error: "API key not found"
**Solution**:
1. Check Settings > Secrets
2. Make sure format is: `FMP_API_KEY = "your_key"`
3. No spaces before FMP_API_KEY
4. Key must be in quotes

### App is too slow
**Solution**: Reduce parameters:
- Population: 30-50
- Generations: 20-30
- Date range: 2-3 years max

### "Rate limit exceeded"
**Solution**: Free FMP accounts have 250 requests/minute. Wait 1 minute and try again.

---

## 📊 What to Expect

### Good Results (Strategy is promising)
```
Train Fitness:      0.75
Validation Fitness: 0.62  (degradation ~17%)
Test Fitness:       0.58  (degradation ~6%)

Sharpe Ratio:       1.5
Max Drawdown:       -12%
Win Rate:           58%
```

### Red Flags (Overfitting)
```
Train Fitness:      0.85
Validation Fitness: 0.35  (degradation >50% ❌)
Test Fitness:       0.20  (degradation >40% ❌)

→ Strategy is overfit, don't use!
```

---

## 💡 Pro Tips

1. **Start Small**: Always test with small parameters first
2. **Check Degradation**: <30% is good, >50% is bad
3. **Multiple Symbols**: Try different stocks to find best fit
4. **Re-optimize**: Markets change, run every 6-12 months
5. **Paper Trade**: NEVER go live without paper trading first!

---

## 🔗 Resources

- **Full Guide**: [STREAMLIT_SETUP.md](STREAMLIT_SETUP.md)
- **Documentation**: [README.md](README.md)
- **FMP Docs**: https://financialmodelingprep.com/developer/docs/

---

## 🆘 Still Stuck?

1. Check the **error message** in the app (it's usually helpful!)
2. Review **Streamlit Cloud logs** (click "Manage app" > "Logs")
3. Make sure all files are in the repo (see file structure below)

### Required File Structure
```
evolution_analysis/
├── app.py              ← MUST BE HERE
├── requirements.txt    ← MUST BE HERE
├── packages.txt        ← MUST BE HERE (for TA-Lib)
├── config/
│   └── config.yaml
└── src/
    ├── __init__.py     ← MUST BE HERE
    ├── data/
    │   └── __init__.py ← MUST BE HERE
    ├── indicators/
    │   └── __init__.py ← MUST BE HERE
    ├── backtesting/
    │   └── __init__.py ← MUST BE HERE
    ├── genetic/
    │   └── __init__.py ← MUST BE HERE
    ├── validation/
    │   └── __init__.py ← MUST BE HERE
    └── utils/
        └── __init__.py ← MUST BE HERE
```

---

**You're all set! Happy optimizing! 🚀📈**
