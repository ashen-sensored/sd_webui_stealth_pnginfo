from modules import script_callbacks, shared, generation_parameters_copypaste
from modules.script_callbacks import ImageSaveParams
import gradio as gr
from modules import images
from PIL import Image
from gradio import media_data, processing_utils, utils
import PIL
import warnings

def add_stealth_pnginfo(params:ImageSaveParams):
    stealth_pnginfo_enabled = shared.opts.data.get("stealth_pnginfo", True)
    if not stealth_pnginfo_enabled:
        return
    if not params.filename.endswith('.png') or params.pnginfo is None:
        return
    source_img = params.image
    width, height = source_img.size
    source_img.putalpha(255)
    pixels = params.image.load()
    str_parameters = params.pnginfo['parameters']
    # prepend signature
    signature_str = 'stealth_pnginfo'

    binary_signature = ''.join(format(byte, '08b') for byte in signature_str.encode('utf-8'))



    binary_param = ''.join(format(byte, '08b') for byte in str_parameters.encode('utf-8'))

    # prepend length of parameters, padded to 32 digits
    param_len = len(binary_param)
    binary_param_len = format(param_len, '032b')

    binary_data = binary_signature + binary_param_len + binary_param
    index = 0
    for x in range(width):
        for y in range(height):
            if index < len(binary_data):
                r, g, b, a = pixels[x, y]

                # Modify the alpha value's least significant bit
                a = (a & ~1) | int(binary_data[index])

                pixels[x, y] = (r, g, b, a)
                index += 1
            else:
                break



    # for k, v in params.pnginfo.items():
    #     pnginfo_data.add_text(k, str(v))


    pass

original_read_info_from_image = images.read_info_from_image

def read_info_from_image_stealth(image):
    geninfo, items = original_read_info_from_image(image)

    # trying to read stealth pnginfo
    width, height = image.size
    pixels = image.load()

    binary_data = ''
    buffer = ''
    index = 0
    sig_confirmed = False
    confirming_signature = True
    reading_param_len = False
    reading_param = False
    read_end = False
    for x in range(width):
        for y in range(height):
            r, g, b, a = pixels[x, y]
            buffer += str(a & 1)
            pixels[x, y] = (r, g, b, 0)
            if confirming_signature:
                if index == len('stealth_pnginfo') * 8 - 1:
                    if buffer == ''.join(format(byte, '08b') for byte in 'stealth_pnginfo'.encode('utf-8')):
                        confirming_signature = False
                        sig_confirmed = True
                        reading_param_len = True
                        buffer = ''
                        index = 0
                    else:
                        read_end = True
                        break
            elif reading_param_len:
                if index == 32:
                    param_len = int(buffer, 2)
                    reading_param_len = False
                    reading_param = True
                    buffer = ''
                    index = 0
            elif reading_param:
                if index == param_len:
                    binary_data = buffer
                    read_end = True
                    break
            else:
                # impossible
                read_end = True
                break

            index += 1
        if read_end:
            break

    if sig_confirmed and binary_data != '':
        # Convert binary string to UTF-8 encoded text
        decoded_data = bytearray(int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8)).decode('utf-8',errors='ignore')

        geninfo = decoded_data

    return geninfo, items

images.read_info_from_image = read_info_from_image_stealth

def send_rgb_image_and_dimension(x):
    if isinstance(x, Image.Image):
        img = x
        if img.mode == 'RGBA':
            img = img.convert('RGB')
    else:
        img = generation_parameters_copypaste.image_from_url_text(x)
        if img.mode == 'RGBA':
            img = img.convert('RGB')


    if shared.opts.send_size and isinstance(img, Image.Image):
        w = img.width
        h = img.height
    else:
        w = gr.update()
        h = gr.update()

    return img, w, h

generation_parameters_copypaste.send_image_and_dimensions = send_rgb_image_and_dimension


def on_ui_settings():
    section = ('stealth_pnginfo', "Stealth PNGinfo")
    shared.opts.add_option("stealth_pnginfo", shared.OptionInfo(
        True, "Save Stealth PNGinfo", gr.Checkbox, {"interactive": True}, section=section))


def custom_image_preprocess(
        self, x
    ):
    if x is None:
        return x

    mask = ""
    if self.tool == "sketch" and self.source in ["upload", "webcam"]:
        assert isinstance(x, dict)
        x, mask = x["image"], x["mask"]


    assert isinstance(x, str)
    im = processing_utils.decode_base64_to_image(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = im.convert(self.image_mode)
    if self.shape is not None:
        im = processing_utils.resize_and_crop(im, self.shape)
    if self.invert_colors:
        im = PIL.ImageOps.invert(im)
    if (
            self.source == "webcam"
            and self.mirror_webcam is True
            and self.tool != "color-sketch"
    ):
        im = PIL.ImageOps.mirror(im)

    if self.tool == "sketch" and self.source in ["upload", "webcam"]:
        mask_im = None
        if mask is not None:
            mask_im = processing_utils.decode_base64_to_image(mask)


        return {
            "image": self._format_image(im),
            "mask": self._format_image(mask_im),
        }

    return self._format_image(im)
def on_after_component_change_pnginfo_image_mode(component, **_kwargs):
    if type(component) is gr.State:
        return
    if type(component) is gr.Image and component.elem_id == 'pnginfo_image':
        component.image_mode = 'RGBA'

    def set_alpha_channel_to_zero(image):
        width, height = image.size
        pixels = image.load()

        for x in range(width):
            for y in range(height):
                r, g, b, a = pixels[x, y]
                pixels[x, y] = (r, g, b, 0)
    def clear_alpha(param):
        print('clear_alpha called')
        output_image = param['image'].convert('RGB')
        return output_image
        # set_alpha_channel_to_zero(input)
        # return input

    if type(component) is gr.Image and component.elem_id == 'img2maskimg':

        component.upload(clear_alpha, component, component)
        component.preprocess = custom_image_preprocess.__get__(component, gr.Image)



script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(add_stealth_pnginfo)
script_callbacks.on_after_component(on_after_component_change_pnginfo_image_mode)