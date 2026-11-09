# Reverted: Chen et al. 2019 (DOI 10.1148/ryai.2019190012) is not an image encoder.
# That paper uses multiscale functional brain connectome data (atlas ROI connectivity
# matrices: AAL 90x90, CC200 190x190, CC400 351x351) plus PCD as input to a
# multichannel DNN. For this codebase it belongs on the ROI/connectivity side, not
# the image branch. Do not use this file as an image encoder.
