nvidia-docker run \
     -it --rm -p5555:8888 \
     -v "${PWD}:/app:rw" \
     -v "/home/raj/github/tcga_cancer_ml_classifier/python:/data/code:rw" \
     -v "/home/raj/github/tcga_cancer_ml_classifier/python/results:/data/output/results:rw" \
     -v "/home/raj/tcgaImageSet_4class_256/train:/data/train:rw" \
     -v "/home/raj/tcgaImageSet_4class_256/test:/data/test:rw" \
     fgiuste/neuroml:py3
