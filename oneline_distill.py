import argparse
import pickle

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import rays_generator
import matplotlib.pyplot as plt 
import copy

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--no_distill', action='store_true',
                    help='Instead of using proposal network, use the corse network.')
parser.add_argument('--record', action='store_true',
                    help='Record the plot')
parser.add_argument('--iter_n', type=int, default=1500)                    

args = parser.parse_args()

device = torch.device(args.device)
L = 32

def set_limit(axs, smoothed_max_value, max_value, min_value=0.01, alpha=0.9):
    if smoothed_max_value is None:
        smoothed_max_value = max_value
    else:
        smoothed_max_value = alpha * smoothed_max_value + (1 - alpha) * max_value
    axs.set_ylim(-0.01, smoothed_max_value)
    return smoothed_max_value

def cast_to_device(input_list):
    out = []
    for x in input_list:
        if isinstance(x, torch.Tensor):
            out.append(x.to(device))
        else:
            out.append(x)
    return out

def pos_encoding(x):
    x = x.reshape(-1, 1)
    l = L // 2 
    pos1 = torch.sin(2 ** torch.arange(l).reshape(1, -1).to(x.device) * torch.pi * x)
    pos2 = torch.cos(2 ** torch.arange(l).reshape(1, -1).to(x.device) * torch.pi * x)
    return torch.cat([pos1, pos2], dim=1)

def sample_according_to_hist(w, x, sample_n, delta, directions, origins):
    n_rays = x.shape[0]
    # NOTE: w already normalized ? 
    #cdf = cdf / cdf.sum(dim=1, keepdims=True)
    cdf = w / w.sum(dim=1, keepdims=True)
    cdf = torch.cumsum(cdf, dim=-1)

    r = torch.rand(size=(n_rays, sample_n)).unsqueeze(dim=-1).to(device)
    cdf = cdf.unsqueeze(dim=1)
    idx = torch.argmax(((cdf - r) > 0).long(), -1)
    batch_index = torch.arange(n_rays).unsqueeze(dim=-1).expand(n_rays, sample_n)
    sample = x[batch_index, idx] + delta[batch_index, idx] *\
                                   torch.rand(size=(n_rays, sample_n)).to(device) *\
                                   directions.unsqueeze(dim=-1)
    sample_sorted = sample.clone()
    sample_sorted[directions == 1, :] = torch.sort(sample, dim=1).values[directions == 1, :]
    sample_sorted[directions == -1, :] = torch.sort(sample, dim=1, descending=True).values[directions == -1, :]
    sample = sample_sorted
    sample = torch.cat([origins, sample], dim=1)
    deltas = torch.abs(sample[:, 1:] - sample[:, :-1])
    return sample[:, :-1].detach(), deltas.detach()

def density_to_w(d, deltas):
    """ 
       a. Assume color value equal to 1
       b. Correpond to equation (3),(4)
    """
    if isinstance(deltas, float):
        deltas = torch.ones_like(d) * deltas
    else:
        deltas = deltas.reshape(d.shape)

    alpha = 1. - torch.exp(-d * deltas)
    trans = torch.cumprod(1 - alpha + 1e-10, -1)
    w = alpha * trans
    w = w / (w.sum(dim=1, keepdim=True) + 1e-5) 
    return w, trans

def loss_prop_fn(w1, t1, w2, t2, deltas_1, deltas_2, directions):
    """
        1 is the proposal interlval and weights
        2 is the more densely sampled points
    """
    loss_prop = 0
    t1 = t1.detach()
    t2 = t2.detach()
    w2 = w2.detach()
    #t2 = t2.flatten()
    #t1 = t1.flatten()
    #w1 = w1.flatten()
    #w2 = w2.flatten()

    rays_n = t1.shape[0] 
    sample_n = t2.shape[1]

    for l in range(rays_n):
        t2_s = t2[l].unsqueeze(dim=0)
        t2_e = (t2[l] + deltas_2[l] * directions[l]).unsqueeze(dim=0)
        w2_ = w2[l]

        #for i, w2_ in enumerate(w2[l]):
        #    t2_s = t2[l][i]
        #    t2_e = t2_s + deltas_2[l][i] * directions[l]
        t1_s = t1[l].unsqueeze(dim=1)
        t1_e = (t1[l] + deltas_1[l] * directions[l]).unsqueeze(dim=1)

        #for j, t1_ in enumerate(t1[l]):
        #    t1_s = t1_
        #    t1_e = t1_s + deltas_1[l][j] * directions[l]
            #if ((t1_ < t2_s and t1_ + deltas_1[l][j] > t2_s) or
            #   (t1_ < t2_e and t1_ + deltas_1[l][j] > t2_e)):

        compare_mat = torch.logical_or((t1_s - t2_s) * (t1_s - t2_e) < 0, (t1_e - t2_s) * (t1_e - t2_e) < 0)
        index1, index2 = torch.where(compare_mat)
        bound = torch.zeros(sample_n).to(device)
        bound[index2] += w1[l][index1]

        #if (t1_s - t2_s) * (t1_s - t2_e) < 0 or (t1_e - t2_s) * (t1_e - t2_e) < 0:
        #    bound_ += w1[l][j]
        loss_prop += (1 / (w2_ + 1e-8) * torch.clip(w2_ - bound, min=0) ** 2).sum()

    return loss_prop


# Ground truth density and sample parameter
x_range = (0, 10)
objects = torch.Tensor([[2, 4], [6, 8]])
N_rays = 100
N_samples = 200
divide_factor = 3


# Proposal MLP
mlp_a = nn.Sequential(nn.Linear(L, 64),
                      nn.ReLU(),
                      nn.Linear(64, 1),
                      nn.ReLU()
                      ).to(device)
# NeRF MLP
mlp_b = nn.Sequential(nn.Linear(L, 64),
                      nn.ReLU(),
                      nn.Linear(64, 64),
                      nn.ReLU(),
                      nn.Linear(64, 1),
                      nn.ReLU()
                      ).to(device)

init_lr = 0.001
optim = torch.optim.AdamW(list(mlp_b.parameters()) + list(mlp_a.parameters()),
                         betas=(0.9, 0.999), eps=1e-6,
                         lr=init_lr)

plt.figure(figsize=(10,  60))
plt.ion()
fig, axs = plt.subplots(5)

# Plot objects
objects_ = objects.numpy().tolist()
for num_plot in range(4):
    for obj in objects:
        axs[num_plot].axvspan(
            obj[0], obj[1], color="red", alpha=0.1
        )

axs[0].title.set_text('Proposal w')
axs[1].title.set_text('NeRF w')
axs[2].title.set_text('NeRF density')
axs[3].title.set_text('NeRF transmittance')
axs[4].title.set_text('total mse depth loss (smoothed)')
axs[0].set_xlim([0, 10])
axs[1].set_xlim([0, 10])
axs[2].set_xlim([0, 10])
axs[3].set_xlim([0, 10])
axs[4].set_xlim([0, 1500])
plt1, = axs[0].plot([],
                  [],
                  marker='o',
                  color = 'r',
                  )
plt2, = axs[1].plot([],
                   [],
                   color='g'
                   )
plt3, = axs[2].plot([],
                   [],
                   marker='o',
                   color='b'
                   )
plt4, = axs[3].plot([],
                   [],
                   marker='o',
                   color='b'
                   )


plt5, = axs[4].plot([],
                    [],
                   )

start_time = time.time()
loss_collect = []
smoothed_loss = []
max_values = [None, None, None, None]
last = None

if args.record:
    recorded_plot = []
if args.no_distill:
    additional_title = 'No distill'
else:
    additional_title = ''

for iter_i in range(args.iter_n):
    # Sample data inference
    origins, x, delta, depths, first_surface, directions = cast_to_device(rays_generator(x_range,
                                                                          objects,
                                                                          N_rays,
                                                                          N_samples // divide_factor))
    # Proposal inference
    d_a = mlp_a(pos_encoding(x)).reshape(N_rays, N_samples // divide_factor)
    w_a, _ = density_to_w(d_a, delta)
    if args.no_distill:
        local_x = torch.abs(x - origins)
        final_a = torch.sum(w_a * local_x, dim=1)
        loss_a = F.mse_loss(final_a, depths)


    # NeRF inference
    sample_bx, deltas_b = sample_according_to_hist(w_a, x, N_samples, delta, directions, origins)
    d_b = mlp_b(pos_encoding(sample_bx)).reshape(N_rays, N_samples) # density b
    w_b, trans_b = density_to_w(d_b, deltas_b)
    local_x = torch.abs(sample_bx - origins)
    final_b = torch.sum(w_b * local_x, dim=1)

    # Proposal loss
    loss = F.mse_loss(final_b, depths)
    loss_collect.append(loss.detach().cpu().item())

    # Collect loss for visualization
    if last is None:
        last = loss_collect[-1]
    point = loss_collect[-1]
    smoothed_loss.append(0.99 * last + (1 - 0.99) * point)
    last = smoothed_loss[-1]

    if not args.no_distill:
        total_loss = loss_prop_fn(w_a, x, w_b, sample_bx, delta, deltas_b, directions) + loss
    else:
        total_loss = loss + loss_a
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(mlp_b.parameters()) + list(mlp_a.parameters()), 1e-3)
    optim.step()
    optim.zero_grad()

    # Plot (reset bar)
    w_a_array = w_a[0].detach().cpu().numpy()
    plt1.set_data(x[0].detach().cpu().numpy(), w_a_array)
                  
    w_b_array = w_b[0].detach().cpu().numpy().flatten()
    plt2.set_data(sample_bx[0].detach().cpu().numpy().flatten(), w_b_array)

    d_b_array = d_b[0].detach().cpu().numpy().flatten()
    plt3.set_data(sample_bx[0].detach().cpu().numpy().flatten(), d_b_array)

    trans_b_array = trans_b[0].detach().cpu().numpy().flatten()
    plt4.set_data(sample_bx[0].detach().cpu().numpy().flatten(), trans_b_array)

    # Plot loss
    plt5.set_data(np.arange(iter_i + 1), smoothed_loss)

    max_values[0] = set_limit(axs[0], max_values[0], np.max(w_a_array) * 1.2)
    max_values[1] = set_limit(axs[1], max_values[1], np.max(w_b_array) * 1.2)
    max_values[2] = set_limit(axs[2], max_values[2], np.max(d_b_array) * 1.2)
    max_values[3] = set_limit(axs[3], max_values[3], np.max(trans_b_array) * 1.2)

    axs[4].set_ylim(-0.01, max(smoothed_loss[-200:]) * 1.2)
    axs[4].set_xlim(max(0, iter_i - 200), iter_i)

    fig.suptitle(f'{additional_title} Ray shoots from 0.5: iter {iter_i}')
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    if iter_i > 500:
        for g in optim.param_groups:
            g['lr'] = 1 / (np.clip((iter_i - 500) / 500, 0, 1) * 100) * init_lr

    if args.record:
        recorded_plot.append(plt.gcf())

end_time = time.time()
print(f'final_time {end_time - start_time}')

if args.record:
    with open('recorded_plot.pickle', 'wb') as f:
        pickle.dump(recorded_plot, f)
