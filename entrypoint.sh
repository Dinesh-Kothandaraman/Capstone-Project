#!/bin/bash
if [ "$RUN_STREAMLIT" = "true" ]; then
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
else
    uvicorn app:app --host 127.0.0.1 --port 8000
fi
