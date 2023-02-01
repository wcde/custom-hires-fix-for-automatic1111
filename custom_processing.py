import numpy as np
import torch
from PIL import Image
import modules.images as images

from modules.shared import opts
from resize_right import resize, interp_methods
from modules import processing, sd_samplers, shared, devices

opt_C = 4
opt_f = 8


class SDProcessing(processing.StableDiffusionProcessingTxt2Img):
    def __init__(self, p: processing.StableDiffusionProcessingTxt2Img, hr_passes, hr_custom_upscaler, cfg_per_pass):
        super().__init__(sd_model=shared.sd_model, outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
                         outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids, prompt=p.prompt,
                         negative_prompt=p.negative_prompt, seed=p.seed, subseed=p.subseed,
                         subseed_strength=p.subseed_strength,
                         seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w,
                         sampler_name=p.sampler,
                         batch_size=p.batch_size, n_iter=p.n_iter, steps=p.steps, cfg_scale=p.cfg_scale,
                         width=p.width, height=p.height, restore_faces=p.restore_faces, tiling=p.tiling,
                         enable_hr=p.enable_hr, hr_upscaler=p.hr_upscaler, hr_second_pass_steps=p.hr_second_pass_steps,
                         denoising_strength=p.denoising_strength, hr_scale=p.hr_scale)
        self.hr_passes = hr_passes
        self.hr_custom_upscaler = hr_custom_upscaler
        self.cfg_per_pass = cfg_per_pass
        self.hr_steps = p.hr_second_pass_steps

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

        add_target_width = (self.hr_upscale_to_x - self.width) / self.hr_passes
        add_target_height = (self.hr_upscale_to_y - self.height) / self.hr_passes

        for current_pass in range(1, self.hr_passes + 1):
            self.hr_second_pass_steps = int(self.hr_steps + self.hr_steps / current_pass)
            self.cfg_scale = self.cfg_scale + (2 * self.cfg_per_pass)

            if current_pass == self.hr_passes:
                target_width = self.hr_upscale_to_x
                target_height = self.hr_upscale_to_y
            else:
                target_width = int(self.width + add_target_width * current_pass)
                target_height = int(self.height + add_target_height * current_pass)

            scale = target_width / max(self.seed_resize_from_w, self.width)

            if latent_scale_mode is None:
                decoded_samples = processing.decode_first_stage(self.sd_model, samples)
                lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

                batch_images = []
                for i, x_sample in enumerate(lowres_samples):
                    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(np.uint8)
                    image = Image.fromarray(x_sample)

                    image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)
                    image = np.array(image).astype(np.float32) / 255.0
                    image = np.moveaxis(image, 2, 0)
                    batch_images.append(image)

                decoded_samples = torch.from_numpy(np.array(batch_images))
                decoded_samples = decoded_samples.to(shared.device)
                decoded_samples = 2. * decoded_samples - 1.

                samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))

                image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)
            else:
                if self.hr_custom_upscaler == "nearest-exact":
                    samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f),
                                                              mode='nearest-exact')
                else:
                    samples = resize(samples, scale_factors=(scale, scale),
                                     interp_method=getattr(interp_methods, self.hr_custom_upscaler))

                if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                    image_conditioning = self.img2img_image_conditioning(processing.decode_first_stage(self.sd_model,
                                                                                                       samples), samples)
                else:
                    image_conditioning = self.txt2img_image_conditioning(samples)

            shared.state.nextjob()
            if current_pass == 1:
                self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)

            self.noise = processing.create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds,
                                                          subseed_strength=subseed_strength,
                                                          p=self, seed_resize_from_w=self.seed_resize_from_w,
                                                          seed_resize_from_h=self.seed_resize_from_h)
            x = None
            devices.torch_gc()
            samples = self.sampler.sample_img2img(self, samples, self.noise, conditioning, unconditional_conditioning,
                                                  steps=self.hr_second_pass_steps or self.steps,
                                                  image_conditioning=image_conditioning)

            self.seed_resize_from_w = target_width
            self.seed_resize_from_h = target_height

        return samples
