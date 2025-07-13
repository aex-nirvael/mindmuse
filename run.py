'''
copyright Alex Whelan 2025
code for running the model
'''
### TO-DO ###
# tidy up code
# embed a mirror inside the model during inference
# align the model more closely with StyleGAN2 - i.e. norms, etc.
# look at data augmentations
###

from tqdm import tqdm

import torch
import lpips

from model import MindModel, MindDiscriminator
from data import MindDataLoader, MindDataset, save_images

import warnings
warnings.filterwarnings("ignore", message=".*The parameter 'pretrained' is deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum or `None` for 'weights'.*")


def run():
  ## device
  device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

  print(f"[*] Using device {device}...")
  ## Training parameters
  image_size = 256
  chan_out = 3
  epochs = 100
  batch_size = 8
  save_frequency = 200
  outdir = "experiment_1"
  pixel_weight = 0.5
  percep_weight = 0.1
  disc_step = 3
  z_dim = 512

  ## train_dataset
  dataset = MindDataset(image_size)
  loader = MindDataLoader(dataset, batch_size)

  ## training loop setup
  model = MindModel(chan_out, z_dim, batch_size)
  disc = MindDiscriminator()
  model.to(device)
  disc.to(device)
  optimiserG = torch.optim.Adam(model.parameters(), lr=1e-4)
  optimiserD = torch.optim.Adam(disc.parameters(), lr=1e-5)
  l1_loss = torch.nn.L1Loss(reduction='mean')
  lpips_loss = lpips.LPIPS(net='alex').to(device) # best forward scores
  bce_loss = torch.nn.BCELoss()
  
  print("[*] Start Training")
  step_counter = 0

  for epoch in range(epochs):
    progress_bar = tqdm(loader.data_loader)
    model.train(True)
    disc.train(True)

    for data in progress_bar:
      gt = torch.as_tensor(data, dtype=torch.float).permute(0,3,1,2)
      gt = gt.to(device)

      if step_counter % disc_step == 0:
        # discriminator pass - reals
        disc.zero_grad()
        disc_pred_real = disc(gt)
        real_labels = torch.full_like(disc_pred_real, 0.9)
        disc_loss_real = bce_loss(disc_pred_real, real_labels)

        # discriminator pass - fakes
        with torch.no_grad():
          noise = torch.randn((1,1,1,batch_size*z_dim), device=device)
          pred = model(noise)
        disc_pred_fake = disc(pred.detach())
        fake_labels = torch.full_like(disc_pred_fake, 0.1)
        disc_loss_fake = bce_loss(disc_pred_fake, fake_labels)
        disc_loss = disc_loss_real + disc_loss_fake
        disc_loss.backward()
        optimiserD.step()

      # generator pass
      model.zero_grad()
      pred = model(noise)
      disc_pred = disc(pred)
      gen_loss = bce_loss(disc_pred, real_labels)
      pixel_loss = l1_loss(pred, gt)
      percep_loss = lpips_loss(pred, gt).mean()
      total_loss = (pixel_loss * pixel_weight) + (percep_loss * percep_weight) + gen_loss
      total_loss.backward()
      optimiserG.step()

      step_counter += 1
      progress_bar.set_description(f"Epoch {epoch}, step {step_counter}: l1 = {pixel_loss.item()}, lpips = {percep_loss.item()}, disc = {disc_loss.item()}, gen = {gen_loss.item()}")

      if step_counter % save_frequency == 0:
        save_images(gt, pred, outdir, step_counter)
        torch.save(model.state_dict(), f"{outdir}/latest_model.pt")


if __name__ == "__main__":
  run()
