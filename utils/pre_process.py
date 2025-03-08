import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# 强制关闭warning
import shutup
shutup.please()

# 使用Real-ESRGAN进行超分辨率处理，要pip install realesrgan才能使用
# 模型权重会下载到~/.cache/realesrgan/目录下
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def download_realesrgan_model(model_name='RealESRGAN_x4plus.pth', output_dir=None, use_mirror=False):
    """
    下载Real-ESRGAN模型
    
    参数:
        model_name: 模型名称，可选值:
            - 'RealESRGAN_x4plus.pth' (默认，通用模型)
            - 'RealESRGAN_x4plus_anime_6B.pth' (动漫优化模型)
            - 'realesrgan-ncnn-vulkan-20210603.onnx' (ONNX格式模型)
        output_dir: 输出目录，默认为~/.cache/realesrgan/
    """
    
    import os
    import platform
    
    if output_dir is None:
        output_dir = os.path.join('.cache', 'realesrgan')
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_name)
    
    if os.path.exists(output_path):
        print(f"模型文件 {model_name} 已存在于 {output_path}，跳过下载")
        return output_path
    
    model_urls = {
        'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'RealESRGAN_x4plus_anime_6B.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        'realesrgan-ncnn-vulkan-20210603.onnx': 'https://huggingface.co/spaces/akhaliq/Real-ESRGAN/resolve/main/realesrgan-ncnn-vulkan-20210603.onnx'
    }
    
    if model_name in model_urls:
            url = model_urls[model_name]
    else:
        raise ValueError(f"未知的模型名称: {model_name}. 请选择 {list(model_urls.keys())}")
    
    # 根据操作系统选择下载命令
    system = platform.system()
    
    print(f"开始下载模型 {model_name} 到 {output_path}...")
    
    if system == 'Windows':
        # Windows系统使用PowerShell的Invoke-WebRequest
        cmd = f'powershell -command "Invoke-WebRequest -Uri {url} -OutFile {output_path}"'
    else:
        # Linux/Mac系统使用wget或curl
        if os.system('which wget > /dev/null 2>&1') == 0:
            cmd = f'wget -O {output_path} {url}'
        elif os.system('which curl > /dev/null 2>&1') == 0:
            cmd = f'curl -L {url} -o {output_path}'
        else:
            raise RuntimeError("系统中未找到wget或curl工具，请安装后重试")
    
    # 执行下载命令
    return_code = os.system(cmd)
    
    # 检查下载是否成功
    if return_code != 0:
        raise RuntimeError(f"模型下载失败，返回代码: {return_code}")
    
    print(f"模型下载成功: {output_path}")
    return output_path


class SuperResolutionProcessor:
    """Real-ESRGAN超分辨率处理器"""
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                         num_block=23, num_grow_ch=32, scale=4)
        self.real_esrganer = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            device=self.device
        )

    def process(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = self.real_esrganer.enhance(img)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

class SharpenTextEnhancer:
    """插值+锐化处理器"""
    
    def __init__(self, scale=2):
        self.scale = scale
    
    def process(self, img_path, output_path=None):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        img_denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        
        h, w = img_denoised.shape
        img_sr = cv2.resize(img_denoised, (w * self.scale, h * self.scale), 
                          interpolation=cv2.INTER_CUBIC)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_sr)
        
        kernel = np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
        img_sharpened = cv2.filter2D(img_enhanced, -1, kernel)
        
        if output_path:
            cv2.imwrite(output_path, img_sharpened)
            return True
        else:
            return img_sharpened

# test
if __name__ == '__main__':
    processor = SuperResolutionProcessor(download_realesrgan_model())
    img_path = 'test.png'
    img = processor.process(img_path)
    cv2.imwrite('test_sr.png', img)
