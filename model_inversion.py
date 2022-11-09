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
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import legacy


def size_range(s: str) -> List[int]:
    """Accept a range 'a-c' and return as a list of 2 ints."""
    return [int(v) for v in s.split("-")][::-1]


# ----------------------------------------------------------------------------


def load_gan(gan_pkl: str, scale_type: str, size: List[int], device: str):
    # custom size code from https://github.com/eps696/stylegan2ada/blob/master/src/_genSGAN2.py
    if size:
        size = size_range(size)
        print("render custom size: ", size)
        print("padding method:", scale_type)
        custom = True
    else:
        custom = False

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.size = size
    G_kwargs.scale_type = scale_type

    print('Loading networks from "%s"...' % gan_pkl)
    device = torch.device(device)
    with dnnlib.util.open_url(gan_pkl) as f:
        G = legacy.load_network_pkl(f, custom=custom, **G_kwargs)["G_ema"].to(device)  # type: ignore

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

    if outdir is None:
        ctx.fail("--outdir option is required")
    device = torch.device(device)

    os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)

    # Generate image
    print("Generating image for seed %d" % (seed))
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    noise = "const"
    if latent is not None:
        if not isinstance(latent, torch.Tensor):
            z = torch.from_numpy(latent).to(device)
        else:
            z = latent.to(device)
        noise = "none"
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_view = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    if save is not None:
        img_view.save(f"{outdir}/seed{save}.png")
    if show:
        img_view.show()
    return img


def create_f(target_class, classifier, preprocess, gan, outdir, truncation_psi, device):
    def f(x, save=None, show=False):
        pixels = generate_image(
            G=gan,
            outdir=outdir,
            truncation_psi=truncation_psi,
            device=device,
            latent=x,
            save=save,
            show=show,
        )
        input = preprocess(pixels.unsqueeze(dim=0))
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


def num_grad(f, x, delta=1):
    # todo preprocess x to make it 1D
    grad = np.zeros(len(x))
    a = np.copy(x)
    for i in range(len(x)):
        a[i] = x[i] + delta
        # todo postprocess a to make it ?
        grad[i] = (f(a) - f(x)) / delta
        a[i] -= delta
    return x


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
    "--scale-type",
    type=click.Choice(["pad", "padside", "symm", "symmside"]),
    default="pad",
    help="scaling method for --size",
    required=False,
    show_default=True,
)
@click.option("--size", type=size_range, help="size of output (in format x-y)")
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
def model_inversion(
    ctx: click.Context,
    gan_pkl: Optional[str],
    classifier_pkl: Optional[str],
    truncation_psi: Optional[float],
    outdir: Optional[str],
    scale_type: Optional[str],
    size: Optional[size_range],
    seed: Optional[int],
    count_max: int,
    device: Optional[str],
    target_class: Optional[int],
):
    classifier, preprocess = load_classifier(
        classifier_pkl=classifier_pkl, device=device
    )
    gan = load_gan(gan_pkl=gan_pkl, scale_type=scale_type, size=size, device=device)
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
    x = generate_image(
        G=gan, outdir=outdir, truncation_psi=truncation_psi, seed=seed, device=device
    )
    count = 0
    conf = 0
    while conf < 0.4 or count > count_max:
        conf = f(x, save=str(count), show=True)
        print(f"Iteration {count}, Conf = {conf}")
        grad = num_grad(f, x)
        x += grad
        count += 1

    print(f"Model inversion completed in {count} iterations.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    model_inversion()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
