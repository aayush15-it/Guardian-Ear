#!/bin/bash
# Guardian Ear — Docker Entrypoint
set -e

MODE="${1:-api}"

case "$MODE" in
    api)
        echo "Starting Guardian Ear API server..."
        exec uvicorn api.main:app --host 0.0.0.0 --port 8000
        ;;
    dashboard)
        echo "Starting Guardian Ear Dashboard..."
        exec streamlit run ui/dashboard.py --server.port 8501 --server.address 0.0.0.0
        ;;
    train)
        echo "Starting model training..."
        exec python scripts/train.py "${@:2}"
        ;;
    extract)
        echo "Starting feature extraction..."
        exec python -m src.features.audio_pipeline "${@:2}"
        ;;
    *)
        exec "$@"
        ;;
esac
