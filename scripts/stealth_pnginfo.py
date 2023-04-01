from modules import script_callbacks, shared
from modules.script_callbacks import ImageSaveParams
import gradio as gr
from modules import images

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
    if geninfo is None:
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
                _, _, _, a = pixels[x, y]
                buffer += str(a & 1)
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
        image.convert('RGB')

    return geninfo, items

images.read_info_from_image = read_info_from_image_stealth


def on_ui_settings():
    section = ('stealth_pnginfo', "Stealth PNGinfo")
    shared.opts.add_option("stealth_pnginfo", shared.OptionInfo(
        True, "Save Stealth PNGinfo", gr.Checkbox, {"interactive": True}, section=section))

def on_after_component_change_pnginfo_image_mode(component, **_kwargs):
    if type(component) is gr.State:
        return
    if type(component) is gr.Image and component.elem_id == 'pnginfo_image':
        component.image_mode = 'RGBA'

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(add_stealth_pnginfo)
script_callbacks.on_after_component(on_after_component_change_pnginfo_image_mode)