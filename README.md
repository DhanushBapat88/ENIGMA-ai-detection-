<<<<<<< HEAD
# voice-ai-detector

Repository layout:

- n8n/workflow.json — sample workflow
- ml-service/
  - app.py — minimal Flask prediction service
  - feature_extractor.py — placeholder feature extraction
  - model.pkl — placeholder model file
  - requirements.txt — Python dependencies

Replace `model.pkl` with your trained model and implement real feature extraction.

Quick demo
-
To create a demo model and test the service locally:

PowerShell:
```powershell
python -m pip install -r voice-ai-detector/ml-service/requirements.txt
python voice-ai-detector/ml-service/train_model.py
python voice-ai-detector/ml-service/test_predict.py
```

Start the Flask server:
```powershell
python voice-ai-detector/ml-service/app.py
```

Example `curl` (JSON body expects `audio` to be a list of numbers):
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
  -d '{"audio": [2.5]}

'
```@dhanush
->test_predict file setup is done and working properly
->feature_extract file modification done and ready
->app.py initial setup done 
->feature_extractor is running properly 
->ml model train part is completed but only detects the recordings with wav extension
---@Rajath
->did n8n workflow.
-->converted to json file and pasted there.
-->there was option for database (postgres,MySQL,googlesheet) I choosed googlesheet for simplicity.