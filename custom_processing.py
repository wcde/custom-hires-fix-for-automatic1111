import numpy as np
import torch
import types
from PIL import Image
from tqdm import trange

import modules.images as images

from modules.shared import opts
from resize_right import resize, interp_methods
from modules import processing, sd_samplers, shared, devices

opt_C = 4
opt_f = 8

dpmu_factor: float = 0.85


@torch.no_grad()
def sampler_dpmu(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    last_x = None
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = x if i == 0 else model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if sigmas[i + 1] == 0:
            return torch.lerp(denoised, last_x, 0.5) * dpmu_factor
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * (1 + 1 / (2 * r)) * denoised / 2
        if sigmas[i + 2] == 0:
            last_x = x
    return x


class SDProcessing(processing.StableDiffusionProcessingTxt2Img):
    def __init__(self, p: processing.StableDiffusionProcessingTxt2Img, first_upscaler, second_upscaler):
        super().__init__(sd_model=shared.sd_model, outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
                         outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids, prompt=p.prompt,
                         negative_prompt=p.negative_prompt, seed=p.seed, subseed=p.subseed,
                         subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w,
                         sampler_name=p.sampler_name,
                         batch_size=p.batch_size, n_iter=p.n_iter, steps=p.steps, cfg_scale=p.cfg_scale,
                         width=p.width, height=p.height, restore_faces=p.restore_faces, tiling=p.tiling,
                         enable_hr=p.enable_hr, hr_upscaler=p.hr_upscaler, hr_second_pass_steps=p.hr_second_pass_steps,
                         denoising_strength=p.denoising_strength, hr_scale=p.hr_scale)
        self.hr_resize_x = p.hr_resize_x
        self.hr_resize_y = p.hr_resize_y
        self.hr_steps = p.hr_second_pass_steps
        self.pass_num = 2 if first_upscaler != 'None' and second_upscaler != 'None' else 1
        self.first_upscaler = first_upscaler
        self.second_upscaler = second_upscaler
        self.cfg_per_pass = 0
        self.init(None, None, None)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        latent_scale_mode = shared.latent_upscale_modes.get(self.hr_upscaler, None) \
            if self.hr_upscaler is not None else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "nearest")

        x = processing.create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds,
                                             subseeds=subseeds,
                                             subseed_strength=self.subseed_strength,
                                             seed_resize_from_h=self.seed_resize_from_h,
                                             seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning,
                                      image_conditioning=self.txt2img_image_conditioning(x))

        if not self.enable_hr:
            return samples
        add_target_width = (self.hr_upscale_to_x - self.width) / self.pass_num
        add_target_height = (self.hr_upscale_to_y - self.height) / self.pass_num

        for stage in range(1, self.pass_num + 1):
            self.hr_second_pass_steps = self.hr_steps + max(self.steps - self.hr_steps, 0) // 2 if stage == 1 and self.pass_num != 1 else self.hr_steps
            self.cfg_scale = self.cfg_scale + self.cfg_per_pass

            if stage == self.pass_num:
                target_width = self.hr_upscale_to_x
                target_height = self.hr_upscale_to_y
            else:
                target_width = int(self.width + add_target_width * stage)
                target_height = int(self.height + add_target_height * stage)

            scale = target_width / max(self.seed_resize_from_w, self.width)

            upscaler = self.first_upscaler if stage == 1 and self.first_upscaler != 'None' else self.second_upscaler
            if upscaler == 'From webui':
                if latent_scale_mode is not None:
                    samples = torch.nn.functional.interpolate(samples,
                                                              size=(target_height // opt_f, target_width // opt_f),
                                                              mode=latent_scale_mode["mode"],
                                                              antialias=latent_scale_mode["antialias"])
                    if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                        image_conditioning = self.img2img_image_conditioning(processing.decode_first_stage(self.sd_model, samples),
                                                                             samples)
                    else:
                        image_conditioning = self.txt2img_image_conditioning(samples)
                else:
                    decoded_samples = processing.decode_first_stage(self.sd_model, samples)
                    lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    batch_images = []
                    for i, x_sample in enumerate(lowres_samples):
                        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                        x_sample = x_sample.astype(np.uint8)
                        image = Image.fromarray(x_sample)

                        image = images.resize_image(0, image, target_width, target_height,
                                                    upscaler_name=self.hr_upscaler)
                        image = np.array(image).astype(np.float32) / 255.0
                        image = np.moveaxis(image, 2, 0)
                        batch_images.append(image)

                    decoded_samples = torch.from_numpy(np.array(batch_images))
                    decoded_samples = decoded_samples.to(shared.device)
                    decoded_samples = 2. * decoded_samples - 1.

                    samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))

                    image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

            else:
                if upscaler == "Latent(nearest-exact)":
                    samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f),
                                                              mode='nearest-exact')
                else:
                    samples = resize(samples, scale_factors=(scale, scale),
                                     interp_method=getattr(interp_methods, 'lanczos2'))

                if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                    image_conditioning = self.img2img_image_conditioning(processing.decode_first_stage(self.sd_model,
                                                                         samples), samples)
                else:
                    image_conditioning = self.txt2img_image_conditioning(samples)
            shared.state.nextjob()
            if self.sampler_name == 'DPMU':
                self.sampler = sd_samplers.create_sampler('DPM++ 2M', self.sd_model)
                self.sampler.func = sampler_dpmu
            else:
                self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
            self.noise = processing.create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds,
                                                          subseed_strength=subseed_strength,
                                                          p=self, seed_resize_from_w=self.seed_resize_from_w,
                                                          seed_resize_from_h=self.seed_resize_from_h)
            x = None
            samples = self.sampler.sample_img2img(self, samples, self.noise, conditioning, unconditional_conditioning,
                                                  steps=self.hr_second_pass_steps or self.steps,
                                                  image_conditioning=image_conditioning)

            devices.torch_gc()
            self.seed_resize_from_w = target_width
            self.seed_resize_from_h = target_height

        return samples
