import math
import kornia
from PIL import Image
from omegaconf import DictConfig
import numpy as np
import torch
from tqdm import trange
from k_diffusion import sampling
import modules.images as images
import utils
from modules import processing, sd_samplers, shared, devices, sd_samplers_kdiffusion, prompt_parser, script_callbacks


_dpmu_step_shift = 2.0
_dpmu_factor = 1.0
_cond = utils.CondCache(prompt_parser.get_multicond_learned_conditioning)
_uncond = utils.CondCache(prompt_parser.get_learned_conditioning)


@torch.no_grad()
def sampler_dpmu(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    last_x = None
    for i in trange(len(sigmas) - 1, disable=disable):
        if shared.state.interrupted:
            callback({'x': x, 'i': len(sigmas) - 1, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': None})
            return x
        denoised = x if i == 0 else model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if sigmas[i + 1] == 0:
            return torch.lerp(denoised, last_x, 0.5) * _dpmu_factor
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * (1 + 1 / (2 * r)) * denoised / _dpmu_step_shift
        if sigmas[i + 2] == 0:
            last_x = x
        torch.clamp(x, -3.0, 3.0)
    return x


def upscale(p: processing.StableDiffusionProcessing, processed: processing.Processed, config: DictConfig):
    ratio = p.width / p.height
    config.width = config.width if config.width > 0 else int(config.height * ratio)
    config.height = config.height if config.height > 0 else int(config.width / ratio)
    with devices.autocast():
        for i in [1, 2]:
            def denoiser_override(n):
                scheduler = config.first_noise_scheduler if i == 1 else config.second_noise_scheduler
                return sampling.get_sigmas_polyexponential(n, 0.01, 15 if scheduler == 'High denoising' else 7, 0.5,
                                                           devices.device)

            def denoise_callback(denoiser_params: script_callbacks.CFGDenoiserParams):
                if denoiser_params.sampling_step > 0:
                    p.cfg_scale = config.orig_cfg

            if config.callback_set is False:
                script_callbacks.on_cfg_denoiser(denoise_callback)
                config.callback_set = True

            if (config.first_sampler if i == 1 else config.second_sampler) == 'DPMU':
                sampler: sd_samplers_kdiffusion.KDiffusionSampler = sd_samplers.create_sampler('DPM++ 2M', shared.sd_model)
                sampler.func = sampler_dpmu
            else:
                sampler = sd_samplers.create_sampler(config.first_sampler if i == 1 else config.second_sampler, shared.sd_model)

            fs_width = (config.width - processed.width) // 2 + processed.width
            fs_height = (config.height - processed.height) // 2 + processed.height

            image = images.resize_image(0, processed.images[0],
                                        fs_width if i == 1 else config.width, fs_height if i == 1 else config.height,
                                        upscaler_name=config.first_upscaler if i == 1 else config.second_upscaler)
            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            decoded_sample = torch.from_numpy(image)
            decoded_sample = decoded_sample.to(shared.device).to(devices.dtype_vae)
            decoded_sample = 2.0 * decoded_sample - 1.0
            samples = shared.sd_model.get_first_stage_encoding(shared.sd_model.encode_first_stage(decoded_sample.unsqueeze(0)))
            image_conditioning = p.img2img_image_conditioning(decoded_sample, samples)
            noise = processing.create_random_tensors(samples.shape[1:], seeds=[processed.seed], subseeds=[processed.subseed],
                                                     subseed_strength=processed.subseed_strength, p=p)

            if (config.first_morphological_noise if i == 1 else config.second_morphological_noise) != 0.0:
                noise_mask = kornia.morphology.gradient(noise, torch.ones(5, 5).to(devices.device))
                noise_mask = kornia.filters.median_blur(noise_mask, (3, 3)) / 5
                noise *= noise_mask * (1 + (config.first_morphological_noise if i == 1
                                            else config.second_morphological_noise))

            steps = int(config.steps if i == 2 else max(((p.steps - config.steps) / 2) + config.steps, config.steps))

            cond = _cond.get_cond([config.prompt if config.prompt != '' else processed.prompt], 100)
            uncond = _uncond.get_cond([config.negative_prompt if config.negative_prompt != '' else processed.negative_prompt],
                                      100)
            p.denoising_strength = config.first_denoise if i == 1 else config.second_denoise
            p.cfg_scale += config.first_cfg if i == 1 else config.second_cfg
            p.sampler_noise_scheduler_override = denoiser_override if (
                config.first_noise_scheduler if i == 1 else config.second_noise_scheduler) != 'Default' else None
            global _dpmu_step_shift
            _dpmu_step_shift = 2.0 if (
                config.first_noise_scheduler if i == 1 else config.second_noise_scheduler
                                             ) == 'Default' else 1.65 + config.dpmu_step_shift
            devices.torch_gc()
            samples = sampler.sample_img2img(p, samples.to(devices.dtype), noise, cond, uncond, steps=steps,
                                                 image_conditioning=image_conditioning).to(devices.dtype_vae)
            devices.torch_gc()
            decoded_sample = processing.decode_first_stage(shared.sd_model, samples)
            if math.isnan(decoded_sample.min()):
                samples = torch.clamp(samples, -config.clamp_vae, config.clamp_vae)
                decoded_sample = processing.decode_first_stage(shared.sd_model, samples)
            decoded_sample = torch.clamp((decoded_sample + 1.0) / 2.0, min=0.0, max=1.0).squeeze()
            x_sample = 255. * np.moveaxis(decoded_sample.cpu().numpy(), 0, 2)
            x_sample = x_sample.astype(np.uint8)
            image = Image.fromarray(x_sample)
            processed.images[0] = image
