# Custom Hires Fix (webui Extension)
## Webui Extension for customizing highres fix and improve details (currently separated from original highres fix)


#### Update 16.10.23:
- added ControlNet support: choose preprocessor/model in CN settings, but don't enable unit
- added Lora support: put Lora in extension prompt to enable Lora only for upscaling, put Lora in negative prompt to disable active Lora

#### Update 02.07.23:
- code rewritten again
- simplified settings
- fixed batch generation and image saving

#### Update 13.06.23:
- added gaussian noise instead of random

#### Update 29.05.23:
- added ToMe optomization in second pass, latest Auto1111 update required, controlled via "Token merging ratio for high-res pass" in settings
- added "Sharp" setting, should be used only with "Smoothness" if image is too blurry

#### Update 12.05.23:
- added smoothness for negative, completely fix ghosting/smears/dirt on flat colors with high denoising

#### Update 02.04.23:
 ###### Don't forget to clear ui-config.json!
- upscale separated from original high-res fix
- now works with img2img
- many fixes

