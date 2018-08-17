nvidia-docker run \
     -it --rm -p1111:8888 \
     -v "${PWD}:/app:rw" \
     -v "/home/raj/github/tcga_cancer_ml_classifier/python:/data/code:rw" \
     -v "/home/raj/github/tcga_cancer_ml_classifier/python/results:/data/output/results:rw" \
     -v "/home/raj/tcga_6class_1024_10k/train:/data/train:rw" \
     -v "/home/raj/tcga_6class_1024_10k/test:/data/test:rw" \
     fgiuste/neuroml:py3
