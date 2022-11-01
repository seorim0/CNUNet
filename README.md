# **Complex Nested U-Net for Speech Enhancement Using Augmented Two-Level Skip Connections**   
   
This is a repo of the paper "Complex Nested U-Net for Speech Enhancement Using Augmented Two-Level Skip Connections", which is submitted to ICASSP2023.   

**Abstract**：Recent DNN-based speech enhancement (SE) models utilize contextual information on multi-time scales to improve the performance of the SE models. This paper proposes a complex nested U-Net (CNUNet) with augmented two-level skip connections (ATLS). The proposed model uses the complex-valued spectrogram as input to estimate the magnitude and phase spectra simultaneously. But the real and imaginary parts of the input spectrogram are concatenated and processed as real-valued features, and the decoder is separated into two paths estimating real and imaginary features, which results in a significant reduction of training parameters. Also, in the proposed model, additional local skip connections are established between the residual U-Nets residing in the complex multi-scale feature extraction (CMSFE) blocks. Experimental results show that the proposed model achieves superior performance in most metrics compared to recently proposed SE models.

## Update:  
* **2022.11.01** Upload codes


## Requirements 
This repo is tested with Ubuntu 20.04, PyTorch 1.9.0, Python3.7, CUDA11.1. For package dependencies, you can install them by:

```
pip install -r requirements.txt    
```   


## Getting started    
1. Install the necessary libraries.   
2. Set directory paths for your dataset. ([options.py](https://github.com/seorim0/CNUNet/blob/main/options.py)) 
```   
# dataset path
noisy_dirs_for_train = '../Dataset/train/noisy/'   
noisy_dirs_for_valid = '../Dataset/valid/noisy/'   
```   
* You need to modify the `find_pair` function in [utils](https://github.com/seorim0/CNUNet/blob/main/utils/progress.py) according to the data file name you have.        
* And if you need to adjust any parameter settings, you can simply change them.   
3. Run [train_interface.py](https://github.com/seorim0/NUNet-TLS/blob/main/train_interface.py)

## Demo
We will update soon.    

 
## References   
**U2-Net: Going deeper with nested u-structure for salient object detection**   
X. Qin, Z. Zhang, C. Huang, M. Dehghan, O. R. Zaiane, and M. Jagersand   
[[paper]](https://www.sciencedirect.com/science/article/pii/S0031320320302077)  [[code]](https://github.com/xuebinqin/U-2-Net)   
**A nested u-net with self-attention and dense connectivity for monaural speech enhancement**   
X. Xiang, X. Zhang, and H. Chen      
[[paper]](https://ieeexplore.ieee.org/abstract/document/9616439)  
**Time-frequency attention for monaural speech enhancement**   
Q. Zhang, Q. Song, Z. Ni, A. Nicolson, and H. Li  
[[paper]](https://arxiv.org/abs/2111.07518)  
**Monoaural Speech Enhancement Using a Nested U-Net withTwo-Level Skip Connections**   
S. Hwang, S. W. Park, and Y. Park   
[[paper]](https://www.isca-speech.org/archive/pdfs/interspeech_2022/hwang22b_interspeech.pdf)  [[code]](https://github.com/seorim0/NUNet-TLS)   


## Contact  
Please contact us if you have any question or suggestion.   
E-mail: allmindfine@yonsei.ac.kr
