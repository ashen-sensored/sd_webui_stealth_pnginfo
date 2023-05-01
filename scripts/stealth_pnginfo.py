from modules import script_callbacks, shared, generation_parameters_copypaste
from modules.script_callbacks import ImageSaveParams
import gradio as gr
from modules import images
from PIL import Image
from gradio import processing_utils
import PIL
import warnings
import gzip


def add_stealth_pnginfo(params: ImageSaveParams):
    stealth_pnginfo_enabled = shared.opts.data.get("stealth_pnginfo", True)
    stealth_pnginfo_mode = shared.opts.data.get('stealth_pnginfo_mode', 'alpha')
    stealth_pnginfo_compressed = shared.opts.data.get("stealth_pnginfo_compression", True)
    if not stealth_pnginfo_enabled:
        return
    if not params.filename.endswith('.png') or params.pnginfo is None:
        return
    if 'parameters' not in params.pnginfo:
        return
    add_data(params, stealth_pnginfo_mode, stealth_pnginfo_compressed)


def prepare_data(params, mode='alpha', compressed=False):
    signature = f"stealth_{'png' if mode == 'alpha' else 'rgb'}{'info' if not compressed else 'comp'}"
    binary_signature = ''.join(format(byte, '08b') for byte in signature.encode('utf-8'))
    param = params.encode('utf-8') if not compressed else gzip.compress(bytes(params, 'utf-8'))
    binary_param = ''.join(format(byte, '08b') for byte in param)
    binary_param_len = format(len(binary_param), '032b')
    return binary_signature + binary_param_len + binary_param


def add_data(params, mode='alpha', compressed=False):
    binary_data = prepare_data(params.pnginfo['parameters'], mode, compressed)
    if mode == 'alpha':
        params.image.putalpha(255)
    width, height = params.image.size
    pixels = params.image.load()
    index = 0
    end_write = False
    for x in range(width):
        for y in range(height):
            if index >= len(binary_data):
                end_write = True
                break
            values = pixels[x, y]
            if mode == 'alpha':
                r, g, b, a = values
            else:
                r, g, b = values
            if mode == 'alpha':
                a = (a & ~1) | int(binary_data[index])
                index += 1
            else:
                r = (r & ~1) | int(binary_data[index])
                if index + 1 < len(binary_data):
                    g = (g & ~1) | int(binary_data[index + 1])
                if index + 2 < len(binary_data):
                    b = (b & ~1) | int(binary_data[index + 2])
                index += 3
            pixels[x, y] = (r, g, b, a) if mode == 'alpha' else (r, g, b)
        if end_write:
            break


def read_info_from_image_stealth(image):
    geninfo, items = original_read_info_from_image(image)
    # possible_sigs = {'stealth_pnginfo', 'stealth_pngcomp', 'stealth_rgbinfo', 'stealth_rgbcomp'}

    # respecting original pnginfo
    if geninfo is not None:
        return geninfo, items

    # trying to read stealth pnginfo
    width, height = image.size
    pixels = image.load()

    has_alpha = True if image.mode == 'RGBA' else False
    mode = None
    compressed = False
    binary_data = ''
    buffer_a = ''
    buffer_rgb = ''
    index_a = 0
    index_rgb = 0
    sig_confirmed = False
    confirming_signature = True
    reading_param_len = False
    reading_param = False
    read_end = False
    for x in range(width):
        for y in range(height):
            if has_alpha:
                r, g, b, a = pixels[x, y]
                buffer_a += str(a & 1)
                index_a += 1
            else:
                r, g, b = pixels[x, y]
            buffer_rgb += str(r & 1)
            buffer_rgb += str(g & 1)
            buffer_rgb += str(b & 1)
            index_rgb += 3
            if confirming_signature:
                if index_a == len('stealth_pnginfo') * 8:
                    decoded_sig = bytearray(int(buffer_a[i:i + 8], 2) for i in
                                            range(0, len(buffer_a), 8)).decode('utf-8', errors='ignore')
                    if decoded_sig in {'stealth_pnginfo', 'stealth_pngcomp'}:
                        confirming_signature = False
                        sig_confirmed = True
                        reading_param_len = True
                        mode = 'alpha'
                        if decoded_sig == 'stealth_pngcomp':
                            compressed = True
                        buffer_a = ''
                        index_a = 0
                    else:
                        read_end = True
                        break
                elif index_rgb == len('stealth_pnginfo') * 8:
                    decoded_sig = bytearray(int(buffer_rgb[i:i + 8], 2) for i in
                                            range(0, len(buffer_rgb), 8)).decode('utf-8', errors='ignore')
                    if decoded_sig in {'stealth_rgbinfo', 'stealth_rgbcomp'}:
                        confirming_signature = False
                        sig_confirmed = True
                        reading_param_len = True
                        mode = 'rgb'
                        if decoded_sig == 'stealth_rgbcomp':
                            compressed = True
                        buffer_rgb = ''
                        index_rgb = 0
            elif reading_param_len:
                if mode == 'alpha':
                    if index_a == 32:
                        param_len = int(buffer_a, 2)
                        reading_param_len = False
                        reading_param = True
                        buffer_a = ''
                        index_a = 0
                else:
                    if index_rgb == 33:
                        pop = buffer_rgb[-1]
                        buffer_rgb = buffer_rgb[:-1]
                        param_len = int(buffer_rgb, 2)
                        reading_param_len = False
                        reading_param = True
                        buffer_rgb = pop
                        index_rgb = 1
            elif reading_param:
                if mode == 'alpha':
                    if index_a == param_len:
                        binary_data = buffer_a
                        read_end = True
                        break
                else:
                    if index_rgb >= param_len:
                        diff = param_len - index_rgb
                        if diff < 0:
                            buffer_rgb = buffer_rgb[:diff]
                        binary_data = buffer_rgb
                        read_end = True
                        break
            else:
                # impossible
                read_end = True
                break
        if read_end:
            break
    if sig_confirmed and binary_data != '':
        # Convert binary string to UTF-8 encoded text
        byte_data = bytearray(int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8))
        try:
            if compressed:
                decoded_data = gzip.decompress(bytes(byte_data)).decode('utf-8')
            else:
                decoded_data = byte_data.decode('utf-8', errors='ignore')
            geninfo = decoded_data
        except:
            pass
    return geninfo, items


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


def on_ui_settings():
    section = ('stealth_pnginfo', "Stealth PNGinfo")
    shared.opts.add_option("stealth_pnginfo", shared.OptionInfo(
        True, "Save Stealth PNGinfo", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("stealth_pnginfo_mode", shared.OptionInfo(
        "alpha", "Stealth PNGinfo mode", gr.Dropdown, {"choices": ["alpha", "rgb"], "interactive": True},
        section=section))
    shared.opts.add_option("stealth_pnginfo_compression", shared.OptionInfo(
        True, "Stealth PNGinfo compression", gr.Checkbox, {"interactive": True}, section=section))


def custom_image_preprocess(self, x):
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


def stealth_resize_image(resize_mode, im, width, height, upscaler_name=None):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """
    # convert to RGB
    if im.mode == 'RGBA':
        im = im.convert('RGB')

    return original_resize_image(resize_mode, im, width, height, upscaler_name)


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
original_read_info_from_image = images.read_info_from_image
images.read_info_from_image = read_info_from_image_stealth
generation_parameters_copypaste.send_image_and_dimensions = send_rgb_image_and_dimension
original_resize_image = images.resize_image
images.resize_image = stealth_resize_image

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(add_stealth_pnginfo)
script_callbacks.on_after_component(on_after_component_change_pnginfo_image_mode)
