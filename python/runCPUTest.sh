docker run \
    --runtime=nvidia \
    --rm \
    -ti \
    -v "${PWD}:/app" \
    tensorflow/tensorflow:latest-gpu \
    python /app/benchmark.py gpu 10000 

