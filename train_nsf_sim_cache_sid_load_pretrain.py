"""modified from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/train_nsf_sim_cache_sid_load_pretrain.py
by karljeon44
"""
import os
import sys
now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
import datetime
from time import sleep, time as ttime

from train import utils
hps = utils.get_hparams()
assert hps.version == 'v2', "ervc-v2 only compatible with V2"
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import shuffle

import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from lib.infer_pack import commons
from train.data_utils import (
  TextAudioLoaderMultiNSFsid,
  TextAudioLoader,
  TextAudioCollateMultiNSFsid,
  TextAudioCollate,
  DistributedBucketSampler,
)
from lib.infer_pack.models import (
  SynthesizerTrnMs768NSFsid as RVC_Model_f0,
  SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
  MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
)
from lib.infer_pack.discriminator import Discriminator
from train.losses import generator_loss, discriminator_loss, feature_loss, kl_loss, MultiResolutionSTFTLoss
from train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from train.process_ckpt import savee

global_step = 0


class EpochRecorder:
  def __init__(self):
    self.last_time = ttime()

  def record(self):
    now_time = ttime()
    elapsed_time = now_time - self.last_time
    self.last_time = now_time
    elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{current_time}] | ({elapsed_time_str})"


def main():
  global global_step
  logger = utils.get_logger(hps.model_dir)
  logger.info(hps)
  writer = SummaryWriter(log_dir=hps.model_dir)
  writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  torch.manual_seed(hps.train.seed)

  if hps.if_f0 == 1:
    train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
  else:
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
    train_dataset,
    hps.train.batch_size * n_gpus,
    # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
    [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
    num_replicas=n_gpus,
    rank=0,
    shuffle=True,
    )
  # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
  # num_workers=8 -> num_workers=4
  if hps.if_f0 == 1:
    collate_fn = TextAudioCollateMultiNSFsid()
  else:
    collate_fn = TextAudioCollate()
  train_loader = DataLoader(
    train_dataset,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn,
    batch_sampler=train_sampler,
    persistent_workers=True,
    prefetch_factor=8,
  )
  if hps.if_f0 == 1:
    net_g = RVC_Model_f0(
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model,
      is_half=hps.train.fp16_run,
      sr=hps.sample_rate,
      bigv=hps.bigv
    )
  else:
    net_g = RVC_Model_nof0(
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model,
      is_half=hps.train.fp16_run,
      bigv=hps.bigv
    )
  if torch.cuda.is_available():
    net_g = net_g.cuda()

  if hps.new_d:
    net_d = Discriminator(use_spectral_norm=hps.model.use_spectral_norm)
  else:
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

  if torch.cuda.is_available():
    net_d = net_d.cuda()

  optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps,)
  optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps,)

  net_g = net_g.to('cuda')
  net_d = net_d.to('cuda')

  try:  # 如果能加载自动resume
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    logger.info("loaded D")
    # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    global_step = (epoch_str - 1) * len(train_loader)
  except:  # 如果首次不能加载，加载pretrain
    epoch_str = 1
    global_step = 0

    if hps.pretrainG != "":
      logger.info("loaded pretrained %s" % hps.pretrainG)
      pg = torch.load(hps.pretrainG, map_location="cpu")["model"]
      if hps.model.spk_embed_dim != 109:
        pg = {k: v for k, v in pg.items() if not k.startswith('emb_g')}
      # if hps.pretrainV != "":
      #   # drop any keys with 'dec' prefix, will print `_IncompatibleKeys` with missing keys
      #   pg = {k:v for k,v in pg.items() if not k.startswith('dec')}
      print(net_g.load_state_dict(pg, strict=False))

    # if hps.pretrainV != "":
    #   logger.info("loaded pretrained %s" % hps.pretrainV)
    #   vg = torch.load(hps.pretrainV, map_location="cpu")["model_g"]
    #   vg = {k:v for k,v in lg.items() if not k.startswith('conv_pre')}
    #   print(net_g.dec.load_state_dict(vg, strict=False))

    if hps.pretrainD != "":
      logger.info("loaded pretrained %s" % hps.pretrainD)
      print(net_d.load_state_dict(torch.load(hps.pretrainD, map_location="cpu")["model"], strict=False))

  # # show model architecture
  # print(net_d)
  # print(net_g)
  print("D Number of Trainable Params:", sum(p.numel() for p in net_d.parameters()))
  print("G Number of Trainable Params:", sum(p.numel() for p in net_g.parameters()))
  print(f"[!] Discriminator init type: {type(net_d)}")

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scaler = GradScaler(enabled=hps.train.fp16_run)


  cache = []
  for epoch in range(epoch_str, hps.train.epochs+1):
    train_and_evaluate(
      epoch,
      hps=hps,
      nets=[net_g, net_d],
      optims=[optim_g, optim_d],
      scaler=scaler,
      loaders=[train_loader, None],
      logger=logger,
      writers=[writer, writer_eval],
      cache=cache,
      new_d=hps.new_d
    )

    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(epoch, hps, nets, optims, scaler, loaders, logger, writers, cache, new_d=False):
  global global_step
  net_g, net_d = nets
  optim_g, optim_d = optims
  train_loader, eval_loader = loaders
  writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)

  net_g.train()
  net_d.train()

  # stft_criterion = MultiResolutionSTFTLoss(net_d.resolutions) if is_new_d else None
  resolutions = [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]
  stft_criterion = MultiResolutionSTFTLoss(resolutions)

  # Prepare data iterator
  if hps.if_cache_data_in_gpu:
    # Use Cache
    data_iterator = cache
    if not cache:
      # Make new cache
      for batch_idx, info in enumerate(train_loader):
        # Unpack
        pitch = pitchf = None
        if hps.if_f0 == 1:
          phone,phone_lengths,pitch,pitchf,spec,spec_lengths,wave,wave_lengths,sid = info
        else:
          phone,phone_lengths,spec,spec_lengths,wave,wave_lengths,sid = info

        # Load on CUDA
        if torch.cuda.is_available():
          phone = phone.cuda()
          phone_lengths = phone_lengths.cuda()
          if hps.if_f0 == 1:
            pitch = pitch.cuda()
            pitchf = pitchf.cuda()
          sid = sid.cuda()
          spec = spec.cuda()
          spec_lengths = spec_lengths.cuda()
          wave = wave.cuda()
          wave_lengths = wave_lengths.cuda()

        # Cache on list
        if hps.if_f0 == 1:
          cache.append((batch_idx, (phone,phone_lengths,pitch,pitchf,spec,spec_lengths,wave,wave_lengths,sid)))
        else:
          cache.append((batch_idx, (phone,phone_lengths,spec,spec_lengths,wave,wave_lengths,sid)))
    else:
      # Load shuffled cache
      shuffle(cache)
  else:
    # Loader
    data_iterator = enumerate(train_loader)

  # Run steps
  epoch_recorder = EpochRecorder()
  for batch_idx, info in data_iterator:
    # Data
    ## Unpack
    pitch = pitchf = None
    if hps.if_f0 == 1:
      phone,phone_lengths,pitch,pitchf,spec,spec_lengths,wave,wave_lengths,sid = info
    else:
      phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
    ## Load on CUDA
    if hps.if_cache_data_in_gpu == False and torch.cuda.is_available():
      phone = phone.cuda()
      phone_lengths = phone_lengths.cuda()
      if hps.if_f0 == 1:
        pitch = pitch.cuda()
        pitchf = pitchf.cuda()
      sid = sid.cuda()
      spec = spec.cuda()
      spec_lengths = spec_lengths.cuda()
      wave = wave.cuda()

    # Calculate
    with autocast(enabled=hps.train.fp16_run):
      if hps.if_f0 == 1:
        y_hat,ids_slice,x_mask,z_mask,(z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
      else:
        y_hat,ids_slice,x_mask,z_mask,(z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, spec, spec_lengths, sid)

      mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
      )
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)

      with autocast(enabled=False):
        y_hat_mel = mel_spectrogram_torch(
          y_hat.float().squeeze(1),
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.hop_length,
          hps.data.win_length,
          hps.data.mel_fmin,
          hps.data.mel_fmax,
        )
      if hps.train.fp16_run:
        y_hat_mel = y_hat_mel.half()
      wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

      # Discriminator
      if new_d:
        # Discriminator Loss
        disc_real, disc_fake = net_d(wave), net_d(y_hat.detach()),
        with autocast(enabled=False):
          loss_disc, losses_disc_r, losses_disc_g = discriminator_loss([x[0] for x in disc_real], [x[0] for x in disc_fake])
          # for (score_fake, _), (score_real,_) in zip(disc_fake, disc_real):
          #   loss_disc += torch.mean(torch.pow(score_real-1., 2))
          #   loss_disc += torch.mean(torch.pow(score_fake, 2))
          # loss_disc = hps.train.c_disc * loss_disc / len(disc_fake)

      else:
        y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
        with autocast(enabled=False):
          loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

    optim_d.zero_grad()
    scaler.scale(loss_disc).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      loss_stft = 0.
      if new_d:
        # Generator
        disc_fake, disc_real = net_d(wave), net_d(y_hat)
        with autocast(enabled=False):
          # 1. Mel Loss
          loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel

          # 2. KL-divergense
          loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

          # new: Multi-Resolution STFT loss
          loss_sc, loss_mag = stft_criterion(y_hat.squeeze(1), wave.squeeze(1))
          loss_stft = (loss_sc + loss_mag) * hps.train.c_stft

          # # 3. Score loss
          # loss_score = 0.
          # with autocast(enabled=False):
          #   for score_fake, _ in disc_fake:
          #     loss_score += torch.mean(torch.pow(score_fake - 1., 2))
          #   loss_score = loss_score / len(disc_fake)

          # 4. Feature loss
          # loss_fm = feature_loss(fmap_r, fmap_g)
          loss_fm = feature_loss([x[1] for x in disc_real], [x[1] for x in disc_fake])

          # 5. Generator loss
          loss_gen, losses_gen = generator_loss([x[0] for x in disc_fake])

          # loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
          # loss_gen_all = loss_mel + loss_kl + loss_score + loss_fm + loss_gen
          loss_gen_all = loss_kl + loss_mel + loss_stft +  loss_fm + loss_gen
      else:
        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        with autocast(enabled=False):
          # # new: Multi-Resolution STFT loss
          # loss_sc, loss_mag = stft_criterion(y_hat.squeeze(1), wave.squeeze(1))
          # loss_stft = (loss_sc + loss_mag) * hps.train.c_stft

          # original losses
          loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
          loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
          loss_fm = feature_loss(fmap_r, fmap_g)
          loss_gen, losses_gen = generator_loss(y_d_hat_g)

          # remove `loss_stft` if keeping the original implementation
          # loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_stft
          loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if global_step % hps.train.log_interval == 0:
      lr = optim_g.param_groups[0]["lr"]
      logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.*batch_idx / len(train_loader)))

      logger.info([global_step, lr])
      if new_d:
        logger.info(f"[Losses] disc={loss_disc:.3f} | loss_gen={loss_gen:.3f} | loss_fm={loss_fm:.3f} | loss_mel={loss_mel:.3f} | loss_stft={loss_stft:.3f} | loss_kl={loss_kl:.3f}")
      else:
        logger.info(f"[Losses] disc={loss_disc:.3f} | loss_gen={loss_gen:.3f} | loss_fm={loss_fm:.3f} | loss_mel={loss_mel:.3f} | loss_kl={loss_kl:.3f}")
      scalar_dict = {
        "loss/g/total": loss_gen_all,
        "loss/d/total": loss_disc,
        "learning_rate": lr,
        "grad_norm/d": grad_norm_d,
        "grad_norm/g": grad_norm_g,
      }
      scalar_dict.update({
        "loss/g/fm": loss_fm,
        "loss/g/mel": loss_mel,
        "loss/g/kl": loss_kl,
        "loss/g/gen": loss_gen,
      })
      if new_d:
        scalar_dict["loss/g/stft"] = loss_stft

      image_dict = {
        "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
        "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
        "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
      }
      utils.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict,)
    global_step += 1

  if epoch % hps.save_every_epoch == 0:
    if hps.if_latest == 0:
      utils.save_checkpoint(net_g,optim_g,hps.train.learning_rate,epoch,os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),)
      utils.save_checkpoint(net_d,optim_d,hps.train.learning_rate,epoch,os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),)
    else:
      utils.save_checkpoint(net_g,optim_g,hps.train.learning_rate,epoch,os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),)
      utils.save_checkpoint(net_d,optim_d,hps.train.learning_rate,epoch,os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),)
    if hps.save_every_weights == "1":
      ckpt = net_g.state_dict()
      logger.info("saving ckpt %s_e%s:%s" % (hps.name,epoch,savee(ckpt,hps.sample_rate,hps.if_f0,hps.name + "_e%s_s%s" % (epoch, global_step),epoch,hps.version,hps)))

  logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
  if epoch >= hps.total_epoch:
    logger.info("Training is done. The program is closed.")

    ckpt = net_g.state_dict()
    logger.info("saving final ckpt:%s" % savee(ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps))
    sleep(1)
    os._exit(2333333)


if __name__ == "__main__":
  torch.multiprocessing.set_start_method("spawn")
  main()
