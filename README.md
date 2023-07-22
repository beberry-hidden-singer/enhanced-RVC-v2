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
* webUI no longer supports training & vocal extraction
  * only inference + timbre fusion
* Distributed Training no longer supported
* removed i18n library with English as the sole display language on webUI
* additional scripts to:
  * train index for each log dir
  * write pre-filelist that can handle multiple speakers
    * in this case speaker id must show at the end of filename preceded by underscore

## NOTE
* currently fpaths to `rmvpe.pt`  and `hubert_base.pt` are hardcoded as `../pretrain/{}.pt`:
  * hubert in [infer-web.py](infer-web.py#L116)
  * hubert in [extract_features.py](extract_features.py#L34)
  * rmvpe in [vc_infer_pipeline.py](lib/vc_infer_pipeline.py#L133)
  * rmvpe in [extract_f0.py](extract_f0.py#L158)

