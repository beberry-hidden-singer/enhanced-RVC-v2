# Enhanced RVC V2
ervc-v2

## Install
* [requirments](./requirements.txt) does not represent the full set of dependencies, but only the most crucial

## Additional Features
* [Multi-Scale STFT Discriminator](lib/model/discriminator.py#L349)
  * according to [this paper](https://arxiv.org/pdf/2306.06814.pdf): `MS-STFT discriminator trains the decoder efficiently and facilitates the synthesis of audio with better quality than the combination of a multi-period discriminator and multi-scale discriminator`
* [Multi-Resolution STFT Loss](lib/model/losses.py#L158)
  * weighted version based on [SpeechEnhancement](https://github.com/Nitin4525/SpeechEnhancement/blob/master/loss.py#L98)
* [Multi-Resolution Discriminator](lib/model/discriminator.py#L203)
  * can be init with pre-trained weights from [Sovits-5.0](https://github.com/PlayVoice/so-vits-svc-5.0/releases/tag/5.0)
* [BigVGAN](lib/model/generator.py#L413)
  * can be init with pre-trained weights from [NSF-BigVGAN](https://github.com/PlayVoice/NSF-BigVGAN/releases/tag/augment)
  * as of Aug 31 2023, inits GeneratorNSF but with all LeakyRelus replaced by Snake
* option to use `crepe`, `mangio-crepe` and `rmvpe` when extracting f0

## Further modifications
* only compatible with v2
* webUI no longer supports training & vocal extraction
  * only inference + timbre fusion
* Distributed Training no longer supported
* removed i18n library with English as the sole display language on webUI
* additional scripts to:
  * train index for each log dir
  * write pre-filelist that can handle multiple speakers
    * in this case speaker id must show at the end of filename preceded by underscore
  * extract and export small model weights in `./weights` dir

## References
* [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
* [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork)
