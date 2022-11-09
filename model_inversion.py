# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
from typing import List, Optional

import click
from tqdm import tqdm
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import legacy


def load_gan(gan_pkl: str, device: str):
    print('Loading networks from "%s"...' % gan_pkl)
    device = torch.device(device)
    with dnnlib.util.open_url(gan_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
    return G


def generate_image(
    G,
    outdir: str,
    truncation_psi: float,
    device: str,
    seed: int = 0,
    latent: torch.Tensor = None,
    save: str = None,
    show: bool = False,
):
    """Generate images using pretrained network pickle."""

    device = torch.device(device)

    os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)

    # Generate image
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    noise = "const"
    if latent is not None:
        if not isinstance(latent, torch.Tensor):
            z = torch.from_numpy(latent).unsqueeze(dim=0).to(device)
        else:
            z = latent.unsqueeze(dim=0).to(device)
        noise = "none"
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_view = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    if save is not None or show:
        print("Generating image for seed %d" % (seed))
    if save is not None:
        img_view.save(f"{outdir}/seed{save}.png")
    if show:
        img_view.show()
    return img, z.squeeze(dim=0)


def create_f(target_class, classifier, preprocess, gan, outdir, truncation_psi, device):
    def f(z, save=None, show=False):
        pixels, z = generate_image(
            G=gan,
            outdir=outdir,
            truncation_psi=truncation_psi,
            device=device,
            latent=z,
            save=save,
            show=show,
        )
        input = preprocess(torch.permute(pixels, (0, 3, 1, 2))).to(device)
        # todo format pixels for input to classifier
        output = classifier(input)
        output = F.softmax(output[0], dim=0)
        return output[target_class].item()

    return f


def load_classifier(classifier_pkl: str, device: str):
    # weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")
    # model = torch.hub.load("pytorch/vision", "resnet50", weights=weights)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    preprocess = weights.transforms()
    # params = torch.load(classifier_pkl, map_location=torch.device(device))
    # model.load_state_dict(params, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, preprocess


def num_grad(f, z, delta=1):
    grad = torch.zeros(len(z), device=z.device)
    a = z.detach().clone()
    for i in tqdm(range(len(z))):
        a[i] = z[i] + delta
        grad[i] = (f(a) - f(z)) / delta
        a[i] -= delta
    return grad


@click.command()
@click.pass_context
@click.option("--gan", "gan_pkl", help="GAN pickle filename", required=True)
@click.option(
    "--classifier",
    "classifier_pkl",
    help="Classifier pickle filename",
    default="",
    show_default=True,
)
@click.option(
    "--trunc",
    "truncation_psi",
    type=float,
    help="Truncation psi",
    default=1,
    show_default=True,
)
@click.option("--outdir", type=str, help="Where to store generated images")
@click.option(
    "--seed", type=int, help="List of random seeds", default=0, show_default=True
)
@click.option(
    "--count_max",
    type=int,
    help="Max number of model inversion iterations",
    default=100,
    show_default=True,
)
@click.option(
    "--device", type=str, help="Torch Device", default="cuda", show_default=True
)
@click.option(
    "--target_class",
    type=int,
    help="Target Class for Model Inversion",
    default=0,
    show_default=True,
)
@click.option(
    "--step_size",
    type=float,
    help="Step size for gradient ascent",
    default=2,
    show_default=True,
)
def model_inversion(
    ctx: click.Context,
    gan_pkl: Optional[str],
    classifier_pkl: Optional[str],
    truncation_psi: Optional[float],
    outdir: Optional[str],
    seed: Optional[int],
    count_max: int,
    device: Optional[str],
    target_class: Optional[int],
    step_size: Optional[float]
):
    classifier, preprocess = load_classifier(
        classifier_pkl=classifier_pkl, device=device
    )
    gan = load_gan(gan_pkl=gan_pkl, device=device)
    f = create_f(
        target_class=target_class,
        classifier=classifier,
        preprocess=preprocess,
        gan=gan,
        outdir=outdir,
        truncation_psi=truncation_psi,
        device=device,
    )

    # numeric black box gradient ascent
    x, z = generate_image(
        G=gan, outdir=outdir, truncation_psi=truncation_psi, seed=seed, device=device
    )
    count = 0
    conf = 0
    while conf < 0.4 or count > count_max:
        conf = f(z, save=str(count), show=True)
        print(f"Iteration {count}, Conf = {conf}")
        grad = num_grad(f, z)
        z += step_size*grad
        count += 1

    print(f"Model inversion completed in {count} iterations.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    model_inversion()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
