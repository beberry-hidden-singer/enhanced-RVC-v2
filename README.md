# Enhanced RVC V2
ervc-v2

## Additional Features
* support for [Multi-Resolution Discriminator](./lib/infer_pack/discriminator.py#L192)
  * can be init with pre-trained weights in [Sovits-5.0](https://github.com/PlayVoice/so-vits-svc-5.0/releases/tag/5.0)
* support for [Multi-Resolution STFT Loss](./train/losses.py#L153)
* support for [BigVGAN](./lib/infer_pack/models.py#L525)


## Further modifications
* only compatible with v2
* Distributed Training no longer supported
* additional scripts to:
  * train index for each log dir
  * write pre-training filelist that can handle multiple speakers
