# Enhanced RVC V2
ervc-v2

## Additional Features
* support for [Multi-Resolution Discriminator](lib/discriminator.py#L201)
  * can be init with pre-trained weights from [Sovits-5.0](https://github.com/PlayVoice/so-vits-svc-5.0/releases/tag/5.0)
* support for [Multi-Resolution STFT Loss](./train/losses.py#L156)
* support for [BigVGAN](lib/generator.py#L413)
  * can be init with pre-trained weights from [NSF-BigVGAN](https://github.com/PlayVoice/NSF-BigVGAN/releases/tag/release)
* option to use `crepe`, `mangio-crepe` and `rmvpe` when extracting f0

## Further modifications
* only compatible with v2
* Distributed Training no longer supported
* cannot be trained on webUI
* additional scripts to:
  * train index for each log dir
  * write pre-training filelist that can handle multiple speakers
