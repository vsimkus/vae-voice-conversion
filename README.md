# Voice Conversion on unaligned data

## Models
* [VAE](https://arxiv.org/abs/1610.04019)
* [VQVAE](https://papers.nips.cc/paper/7210-neural-discrete-representation-learning)
* [JointVAE](https://arxiv.org/abs/1804.00104)

# Dependencies
* PyTorch v1.0.0 or later
* numpy
* pillow
* tqdm

## For extracting WORLD features:
* [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)

## For preprocessing raw audio features
* [torchaudio](https://github.com/pytorch/audio)
    * [libsox v14.3.2 or later](https://anaconda.org/conda-forge/sox)
    * GCC v4.9 or later

# VCTK dataset changes
* Removed the following silent samples before running preprocessing, `p323_424`, `p306_151`, `p351_361`, `p345_292`, `p341_101`, `p306_352`. 


# Team

* Vaidotas Simkus
* Simon Valentin
* Will Greedy
