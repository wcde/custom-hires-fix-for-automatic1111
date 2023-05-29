import os
from os.path import exists
from modules import scripts, shared, prompt_parser
import gradio as gr
import custom_processing
import utils

utils.safe_import('kornia')
utils.safe_import('omegaconf')
utils.safe_import('pathlib')
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

cond = utils.CondCache(prompt_parser.get_multicond_learned_conditioning)
uncond = utils.CondCache(prompt_parser.get_learned_conditioning)
config_path = Path(__file__).parent.resolve() / '../config.yaml'


class CustomHiresFix(scripts.Script):
    def __init__(self):
        super().__init__()
        if not exists(config_path):
            open(config_path, 'w').close()
        self.config: DictConfig = OmegaConf.load(config_path)
        self.config.callback_set = False

    def title(self):
        return "Custom Hires Fix"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label='Custom hires fix', open=False):
            enable = gr.Checkbox(label='Enable extension', value=self.config.get('enable', False))
            with gr.Row():
                width = gr.Slider(minimum=512, maximum=2048, step=8,
                                  label="Upscale width to",
                                  value=self.config.get('width', 1024), allow_flagging='never', show_progress=False)
                height = gr.Slider(minimum=512, maximum=2048, step=8,
                                   label="Upscale height to",
                                   value=self.config.get('height', 0), allow_flagging='never', show_progress=False)
            with gr.Row():
                steps = gr.Slider(minimum=5, maximum=25, step=1,
                                  label="Steps",
                                  value=self.config.get('steps', 12))
                smoothness = gr.Slider(minimum=-1, maximum=3, step=1,
                                        label="Smoothness",
                                        value=self.config.get('smoothness', 1))
            with gr.Row():
                prompt = gr.Textbox(label='Prompt for upscale',
                                    placeholder='Leave empty for using generation prompt',
                                    value=self.config.get('prompt', ''))
            with gr.Row():
                negative_prompt = gr.Textbox(label='Negative prompt for upscale',
                                             placeholder='Leave empty for using generation negative prompt',
                                             value=self.config.get('negative_prompt', ''))
            with gr.Row():
                first_upscaler = gr.Dropdown([*[x.name for x in shared.sd_upscalers
                                                if x.name not in ['None', 'Lanczos', 'Nearest']]],
                                             label='First upscaler',
                                             value=self.config.get('first_upscaler', 'R-ESRGAN 4x+'))
                second_upscaler = gr.Dropdown([*[x.name for x in shared.sd_upscalers
                                                 if x.name not in ['None', 'Lanczos', 'Nearest']]],
                                              label='Second upscaler',
                                              value=self.config.get('second_upscaler', 'R-ESRGAN 4x+'))
            with gr.Row():
                first_cfg = gr.Slider(minimum=0, maximum=10, step=1, label="CFG scale boost (1)",
                                      value=self.config.get('first_cfg', 5))
                second_cfg = gr.Slider(minimum=0, maximum=10, step=1, label="CFG scale boost (2)",
                                       value=self.config.get('second_cfg', 5))
            with gr.Row():
                first_denoise = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, label="Denoise strength (1)",
                                          value=self.config.get('first_denoise', 0.50))
                second_denoise = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, label="Denoise strength (2)",
                                           value=self.config.get('second_denoise', 0.50))
            with gr.Row():
                first_morphological_noise = gr.Slider(minimum=0.0, maximum=2.5, step=0.01,
                                                      label="Morphological noise (1)",
                                                      value=self.config.get('first_morphological_noise', 0.0))
                second_morphological_noise = gr.Slider(minimum=0.0, maximum=2.5, step=0.01,
                                                       label="Morphological noise (2)",
                                                       value=self.config.get('second_morphological_noise', 0.0))
            with gr.Row():
                first_morphological_noise_blur = gr.Slider(minimum=0, maximum=5, step=1,
                                                      label="Morph mask blur (1)",
                                                      value=self.config.get('first_morphological_noise_blur', 3))
                second_morphological_noise_blur = gr.Slider(minimum=0, maximum=5, step=1,
                                                       label="Morph mask blur (2)",
                                                       value=self.config.get('second_morphological_noise_blur', 3))
            with gr.Row():
                first_sampler = gr.Dropdown(['DPM++ 2M', 'DPMU', 'Euler a', 'DPM++ SDE'], label='Sampler (1)',
                                            value=self.config.get('first_sampler', 'DPMU'))
                second_sampler = gr.Dropdown(['DPM++ 2M', 'DPMU', 'Euler a', 'DPM++ SDE'], label='Sampler (2)',
                                             value=self.config.get('second_sampler', 'DPMU'))
            with gr.Row():
                first_noise_scheduler = gr.Dropdown(['High denoising', 'Low denoising', 'Default'],
                                                    label='Noise scheduler (1)',
                                                    value=self.config.get('first_noise_scheduler', 'Low denoising'))
                second_noise_scheduler = gr.Dropdown(['High denoising', 'Low denoising', 'Default'],
                                                     label='Noise scheduler (2)',
                                                     value=self.config.get('second_noise_scheduler', 'Low denoising'))
            with gr.Row():
                dpmu_factor = gr.Slider(minimum=0.6, maximum=1.0, step=0.01,
                                        label="DPMU output factor (color correction)",
                                        value=self.config.get('dpmu_factor', 0.9))
                dpmu_step_shift = gr.Slider(minimum=-0.2, maximum=0.2, step=0.01,
                                            label="DPMU step shift (color correction)",
                                            value=self.config.get('dpmu_step_shift', 0.0))
            with gr.Row():
                clip_skip = gr.Slider(minimum=1, maximum=5, step=1,
                                            label="Clip skip",
                                            value=self.config.get('clip_skip', 2))
                clamp_vae = gr.Slider(minimum=1.0, maximum=10.0, step=1.0, label="Clamp VAE input (NaN VAE fix)", value=3.0)
            sharp = gr.Checkbox(label='Sharp', value=self.config.get('sharp', False))
        if is_img2img:
            width.change(fn=lambda x: gr.update(value=0), inputs=width, outputs=height)
            height.change(fn=lambda x: gr.update(value=0), inputs=height, outputs=width)
        else:
            width.change(fn=lambda x: gr.update(value=0), inputs=width, outputs=height)
            height.change(fn=lambda x: gr.update(value=0), inputs=height, outputs=width)

        ui = [enable, width, height, steps, smoothness, sharp,
              first_upscaler, second_upscaler, first_cfg, second_cfg, first_denoise, second_denoise,
              first_morphological_noise, second_morphological_noise, first_morphological_noise_blur, second_morphological_noise_blur,
              first_sampler, second_sampler, first_noise_scheduler, second_noise_scheduler, dpmu_factor,
              dpmu_step_shift, prompt, negative_prompt, clip_skip, clamp_vae]
        for elem in ui:
            setattr(elem, "do_not_save_to_config", True)
        return ui


    def postprocess(self, p, processed,
                    enable, width, height, steps, smoothness, sharp,
                    first_upscaler, second_upscaler, first_cfg, second_cfg, first_denoise, second_denoise,
                    first_morphological_noise, second_morphological_noise, first_morphological_noise_blur, second_morphological_noise_blur,
                    first_sampler, second_sampler, first_noise_scheduler, second_noise_scheduler, dpmu_factor,
                    dpmu_step_shift, prompt, negative_prompt, clip_skip, clamp_vae
                    ):
        if not enable:
            return processed
        self.config.width = width
        self.config.height = height
        self.config.prompt = prompt
        self.config.negative_prompt = negative_prompt
        self.config.steps = steps
        self.config.smoothness = smoothness
        self.config.sharp = sharp
        self.config.first_cfg = first_cfg
        self.config.second_cfg = second_cfg
        self.config.first_sampler = first_sampler
        self.config.second_sampler = second_sampler
        self.config.first_upscaler = first_upscaler
        self.config.second_upscaler = second_upscaler
        self.config.first_noise_scheduler = first_noise_scheduler
        self.config.second_noise_scheduler = second_noise_scheduler
        self.config.first_denoise = first_denoise
        self.config.second_denoise = second_denoise
        self.config.first_morphological_noise = first_morphological_noise
        self.config.second_morphological_noise = second_morphological_noise
        self.config.first_morphological_noise_blur = first_morphological_noise_blur
        self.config.second_morphological_noise_blur = second_morphological_noise_blur
        self.config.dpmu_step_shift = dpmu_step_shift
        self.config.dpmu_factor = dpmu_factor
        self.config.first_sampler_name = first_sampler
        self.config.clip_skip = clip_skip
        self.config.clamp_vae = clamp_vae
        self.config.orig_cfg = p.cfg_scale
        self.config.callback_set = False
        OmegaConf.save(self.config, config_path)
        custom_processing._config = self.config
        return custom_processing.upscale(p, processed, self.config)
