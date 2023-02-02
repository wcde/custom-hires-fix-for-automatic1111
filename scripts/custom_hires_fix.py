import torch
from modules import scripts, script_callbacks, processing
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
        self.hr_prog = None
        self.force_details = None
        self.cfg_per_step = None
        self.steps = None
        self.pr = None
        self.hr_sampler = None
        self.first_pass_started = True
        self.first_pass_processing = False
        self.second_pass_processing = False
        self.cfg = None
        self.callback_set = False
    
    def title(self):
        return "Custom hires fix"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        with gr.Blocks():
            with gr.Row():
                force_details = gr.Slider(minimum=0, maximum=10, step=1, label="Force details", value=5)
                hr_prog = gr.Checkbox(label="Smooth details", value=True)
                hr_passes = gr.Slider(minimum=1, maximum=3, step=1, label="Passes", value=2)
        with gr.Accordion("Extra", open=False):
            with gr.Row():
                noise_sheduler = gr.Dropdown(['Karras', 'VP', 'Exp'],
                                                label='Noise Scheduler', value='Karras', type='value')
                hr_custom_upscaler = gr.Dropdown(['lanczos2', 'lanczos3', 'nearest-exact'],
                                          label='Latent upscaler', value='lanczos2', type='value')
                beta = gr.Slider(minimum=1, maximum=20, step=1, label="VP Beta", value=10)
            with gr.Row():
                hr_sampler = gr.Dropdown(['DPM++ 2M', 'Euler a', 'Euler', 'DDIM'],
                                         label='Hires Sampler', value='DPM++ 2M', type='value')
                hr_noise_sheduler = gr.Dropdown(['Poly Exp', 'VP', 'Karras'],
                                                label='Hires Noise Scheduler', value='Poly Exp', type='value')
            with gr.Blocks():
                with gr.Row():
                    hr_sigma = gr.Slider(minimum=1, maximum=25, step=1, label="HR Sigma", value=7)
                    hr_rho = gr.Slider(minimum=0.5, maximum=20, step=0.5, label="HR Rho (Karras and Poly)", value=0.5)
                with gr.Row():
                    hr_beta = gr.Slider(minimum=0.5, maximum=20, step=0.5, label="HR Beta max (VP)", value=7)
                    hr_beta_min = gr.Slider(minimum=0.1, maximum=20, step=0.1, label="HR Beta min (VP)", value=0.1)
                with gr.Row():
                    hr_sigma_min = gr.Slider(minimum=0.01, maximum=3, step=0.01, label="HR Sigma min", value=0.01)
                    hr_eps = gr.Slider(minimum=0.001, maximum=0.5, step=0.001, label="HR Esp (VP)", value=0.001)

        return [hr_noise_sheduler, hr_sigma_min, hr_sigma, hr_rho, hr_beta, hr_beta_min, hr_eps, noise_sheduler, beta,
                hr_sampler, hr_passes, hr_custom_upscaler, force_details, hr_prog]
    
    def denoise_callback(self, p: script_callbacks.CFGDenoiserParams):
        if p.sampling_step == (self.steps - 2) and self.first_pass_processing:  # UI bug
            self.pr.sampler_name = self.hr_sampler
        if self.second_pass_processing and self.hr_prog:
            self.pr.cfg_scale = self.pr.cfg_scale - (self.force_details * 0.14)
        if self.second_pass_processing and p.sampling_step != 0 and not self.hr_prog:
            self.pr.cfg_scale = self.cfg


    def run(self, p: processing.StableDiffusionProcessingTxt2Img,
                hr_noise_sheduler, hr_sigma_min, hr_sigma, hr_rho, hr_beta, hr_beta_min, hr_eps, noise_sheduler,
                beta, hr_sampler, hr_passes, hr_custom_upscaler, force_details, hr_prog):
        p = custom_processing.SDProcessing(p, hr_passes=hr_passes, hr_custom_upscaler=hr_custom_upscaler,
                                           cfg_per_pass=force_details)
        self.force_details = force_details
        self.cfg = p.cfg_scale
        self.hr_prog = hr_prog

        def denoiser_override(n):
            if self.first_pass_started:
                self.first_pass_started = False
                self.first_pass_processing = True
                if noise_sheduler == 'Exp':
                    return sampling.get_sigmas_exponential(n, 0.1, 10, self.device)
                if noise_sheduler == 'Karras':
                    return sampling.get_sigmas_karras(n, 0.1, 10, 7, self.device)
                if noise_sheduler == 'VP':
                    return sampling.get_sigmas_vp(n, beta, 0.1, 0.001).to(self.device)
            else:
                self.first_pass_processing = False
                self.second_pass_processing = True
                if hr_noise_sheduler == 'Karras':
                    return sampling.get_sigmas_karras(n, hr_sigma_min, hr_sigma, hr_rho, self.device)
                if hr_noise_sheduler == 'VP':
                    return sampling.get_sigmas_vp(n, hr_beta, hr_beta_min, hr_eps).to(self.device)
                if hr_noise_sheduler == 'Poly Exp':
                    return sampling.get_sigmas_polyexponential(n, hr_sigma_min, hr_sigma, hr_rho, self.device)

        p.sampler_noise_scheduler_override = denoiser_override
        self.steps = p.steps
        self.pr = p
        self.hr_sampler = hr_sampler
        self.first_pass_started = True
        self.first_pass_processing = False
        self.second_pass_processing = False
        if not self.callback_set:
            script_callbacks.on_cfg_denoiser(self.denoise_callback)
            self.callback_set = True

        return processing.process_images(p)
