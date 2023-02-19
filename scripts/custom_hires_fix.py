import torch
from modules import scripts, script_callbacks, processing, shared
import gradio as gr
from k_diffusion import sampling
import custom_processing

try:
    import resize_right
except Exception:
    import pip

    if hasattr(pip, 'main'):
        pip.main(['install', 'resize-right'])
    else:
        pip._internal.main(['install', 'resize-right'])


class CustomHiresFix(scripts.Script):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proc = None
        self.callback_set = False
        self.stage = 'Gen'
        self.last_step = 0
        self.original_denoise = 0.0
        self.original_cfg = 0
        self.first_cfg = 0
        self.second_cfg = 0
        self.first_sampler = ''
        self.second_sampler = ''
        self.first_noise_scheduler = ''
        self.second_noise_scheduler = ''
        self.first_denoise = 0.0
        self.second_denoise = 0.0

    def title(self):
        return "Custom hires fix"

    def show(self, is_img2img):
        if not is_img2img:
            return scripts.AlwaysVisible
        else:
            return False

    def ui(self, is_img2img):
        with gr.Accordion(label='Custom hires fix', open=False):
            with gr.Row():
                first_upscaler = gr.Dropdown(['From webui', 'Latent(lanczos2)', 'Latent(nearest-exact)', 'None'],
                                             label='First upscaler', value='From webui')
                second_upscaler = gr.Dropdown(['From webui', 'Latent(lanczos2)', 'Latent(nearest-exact)', 'None'],
                                              label='Second upscaler', value='Latent(lanczos2)')
            with gr.Row():
                first_cfg = gr.Slider(minimum=0, maximum=10, step=1, label="CFG scale boost (1)", value=3)
                second_cfg = gr.Slider(minimum=0, maximum=10, step=1, label="CFG scale boost (2)", value=5)
            with gr.Row():
                first_denoise = gr.Slider(minimum=-0.3, maximum=0.3, step=0.01, label="Denoise strength shift (1)", value=-0.05)
                second_denoise = gr.Slider(minimum=-0.3, maximum=0.3, step=0.01, label="Denoise strength shift (2)", value=0.0)
            with gr.Row():
                first_sampler = gr.Dropdown(['DPM++ 2M', 'Euler a'], label='Sampler (1)', value='DPM++ 2M')
                second_sampler = gr.Dropdown(['DPM++ 2M', 'Euler a'], label='Sampler (2)', value='DPM++ 2M')
            with gr.Row():
                first_noise_scheduler = gr.Dropdown(['High denoising', 'Low denoising'], label='Noise scheduler (1)', value='High denoising')
                second_noise_scheduler = gr.Dropdown(['High denoising', 'Low denoising'], label='Noise scheduler (2)', value='Low denoising')
            with gr.Row():
                disable = gr.Checkbox(label='Disable extension', value=False)

        return [first_upscaler, second_upscaler, first_cfg, second_cfg, first_denoise, second_denoise,
                first_sampler, second_sampler, first_noise_scheduler, second_noise_scheduler, disable]

    def denoise_callback(self, p: script_callbacks.CFGDenoiserParams):
        def denoiser_override(n):
            scheduler = self.first_noise_scheduler if self.stage == 'Stage 1' else self.second_noise_scheduler
            return sampling.get_sigmas_polyexponential(n, 0.01, 15 if scheduler == 'High denoising' else 7, 0.5, self.device)

        is_last_step = p.sampling_step == p.total_sampling_steps - 2
        is_duplicate = self.last_step == p.sampling_step

        if p.sampling_step != 0 and not is_last_step:
            self.proc.cfg_scale = self.original_cfg - max((self.first_cfg // 3) if self.stage == 'Stage 1' else (self.second_cfg // 3), 3)

        if self.stage == 'Gen' and is_last_step and not is_duplicate:
            self.stage = 'Stage 1'
        elif self.stage == 'Stage 1' and is_last_step and not is_duplicate:
            self.stage = 'Stage 2'
        elif self.stage == 'Stage 2' and is_last_step and not is_duplicate:
            shared.disable_custom_hires_fix = False   # for xyz plot
            self.stage = 'Completed'

        self.proc.sampler_name = self.first_sampler if self.stage == 'Stage 1' else self.second_sampler
        self.proc.denoising_strength = self.original_denoise + (self.first_denoise if self.stage == 'Stage 1' else self.second_denoise)
        self.proc.cfg_per_pass = self.first_cfg if self.stage == 'Stage 1' else self.second_cfg

        if self.stage == 'Completed':
            self.proc.denoising_strength = self.original_denoise
            self.proc.cfg_per_pass = self.original_cfg

        self.last_step = p.sampling_step

    def process(self, p: processing.StableDiffusionProcessingTxt2Img,
                first_upscaler, second_upscaler, first_cfg, second_cfg, first_denoise, second_denoise,
                first_sampler, second_sampler, first_noise_scheduler, second_noise_scheduler, disable):
        if disable or p.denoising_strength == None:
            self.stage = 'Gen'
            return
        if hasattr(shared, 'disable_custom_hires_fix'):   # for xyz plot
            if shared.disable_custom_hires_fix:
                return
        if self.stage == 'Completed' or 'Stage 2':
            self.stage = 'Gen'
        self.first_cfg = first_cfg * 2
        self.second_cfg = second_cfg * 2
        self.first_sampler = first_sampler
        self.second_sampler = second_sampler
        self.first_noise_scheduler = first_noise_scheduler
        self.second_noise_scheduler = second_noise_scheduler
        self.first_denoise = first_denoise
        self.second_denoise = second_denoise
        custom = custom_processing.SDProcessing(p, first_upscaler, second_upscaler)
        p.sample = custom.sample
        self.proc = custom
        if not self.callback_set:
            script_callbacks.on_cfg_denoiser(self.denoise_callback)
            self.original_cfg = p.cfg_scale
            self.original_denoise = p.denoising_strength
            self.stage = 'Gen'
            self.callback_set = True
