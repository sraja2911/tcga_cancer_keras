nvidia-docker run \
     -it --rm -p1111:8888 \
     -v "${PWD}:/app:rw" \
     -v "/home/raj/github/tcga_cancer_ml_classifier/python:/data/code:rw" \
     -v "/home/raj/github/tcga_cancer_ml_classifier/python/results:/data/output/results:rw" \
     -v "/home/raj/tcgaImageSet_1024/train:/data/train:rw" \
     -v "/home/raj/tcgaImageSet_1024/test:/data/test:rw" \
     fgiuste/neuroml:V3
