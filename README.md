# Golf Ball Sorter (POC)
- Run UI: `pip install -r requirements.txt && python webui/main.py`
- Pi client: set SERVER_HOST/PORT/CAMERA_ID then `python capture/pi_client.py`
- Training script skeleton at `ml/train_ballnet.py` (see chat for full loop)
- Docker: `docker build -t ball-sorter . && docker run -p 8000:8000 ball-sorter`
