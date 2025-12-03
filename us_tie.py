import torch
from torch.fft import fft2, ifft2, fftfreq
from PIL import Image
import numpy as np

def load_image(fname, device):
    img = Image.open(fname).convert("L")
    arr = np.array(img) / 255
    return torch.tensor(arr, device=device)

def create_grids(H, W, dx, dy, device='cuda'):
    fx = fftfreq(W, dx, device=device)
    fy = fftfreq(H, dy, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    KX = 2.0 * torch.pi * FX
    KY = 2.0 * torch.pi * FY
    k_sq = KX**2 + KY**2
    k_sq[0, 0] = 1.0
    return KX, KY, k_sq

def gradient(f, KX, KY):
    F_f = fft2(f)
    fdx = ifft2(1j * KX * F_f).real
    fdy = ifft2(1j * KY * F_f).real
    return fdx, fdy

def divergence(fx, fy, KX, KY):
    F_fx = fft2(fx)
    F_fy = fft2(fy)
    div_f = ifft2(1j * KX * F_fx + 1j * KY * F_fy).real
    return div_f

def single_tie(I_0, dIdz, lam, dx, dy, device='cuda'):
    k = 2.0 * torch.pi / lam
    H, W = I_0.shape
    Imax = I_0.max()

    KX, KY, k_sq = create_grids(H, W, dx, dy, device)      
             
    Fphi = -fft2(-k * dIdz / Imax) / k_sq      
    phi = ifft2(Fphi).real      
            
    dphidx, dphidy = gradient(phi, KX, KY)      
           
    dIdz_k = -divergence(I_0 * dphidx, I_0 * dphidy, KX, KY) / k
        
    return phi-phi.mean(), dIdz_k

def us_fft_tie(
    I_0: torch.Tensor,
    dIdz: torch.Tensor,
    lam: float,
    dx: float,
    dy: float,
    max_iter: int = 50,
    tol: float = 1e-3,
    st: float = 0.95,
    device = 'cuda'
) -> torch.Tensor:
    """
    基于迭代求解强度传输方程(TIE)的相位恢复算法

    参数:      
        I_0 : 聚焦平面的二维光强分布，形状为 (H, W)      
        dIdz : 光强在传播方向(z方向)的导数，形状与I_0相同      
        lam : 光的波长，单位应与dx,dy一致      
        dx : x方向的空间采样间隔      
        dy : y方向的空间采样间隔      
        max_iter : 可选，最大迭代次数，默认为50      
        tol : 可选，收敛容差，默认为1e-3      
        st: 可选，收敛停滞阈值，默认为0.95
            当前残差 > st * 上次残差时，认为收敛停滞
        device : 可选，计算设备，默认为'cuda'(GPU)      
        
    返回:      
        恢复的相位分布，形状与I_0相同      
        
    算法原理请参考：      
    On a universal solution to the transport-of-intensity equation      
    DOI: 10.1364/OL.391823      
    """      
    I_0 = I_0.to(device=device)      
    dIdz = dIdz.to(device=device)      
        
    delta_dIdz_k = dIdz.clone()      
    Phi_k = torch.zeros_like(I_0, device=device)      
        
    norm = torch.norm(delta_dIdz_k)      
        
    for k_iter in range(max_iter):      
        phi_k, dIdz_k = single_tie(I_0, delta_dIdz_k, lam, dx, dy, device)      
               
        delta_dIdz_k = delta_dIdz_k - dIdz_k      
        Phi_k = Phi_k + phi_k      
                 
        res_norm = torch.norm(delta_dIdz_k) / torch.norm(dIdz)      
            
        print(f"Iter {k_iter+1}: 相对残差 = {res_norm.item():.4e}")      
                 
        if res_norm < tol:      
            print(f"达到收敛容差，迭代次数 {k_iter+1}")      
            break      
                 
        if res_norm > norm:      
            print(f"残差增加，可能发散，迭代次数 {k_iter+1}")      
            break      

        elif res_norm > st * norm:
            rel_decrease = 100 * (norm - res_norm) / norm
            print(f"收敛停滞（残差下降{rel_decrease:.2f}% < {100*(1-st):.1f}%），停止迭代")
            break

        if k_iter == max_iter - 1:      
            print(f"达到最大迭代次数 {max_iter}")  
                 
        norm = res_norm      
        
    return Phi_k - Phi_k.mean()