import torch
import torch.nn.functional as F
import numpy as np
import h5py

# --- Config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32
ALIGN_CORNERS = False
PADDING_MODE = "reflection" # 'zeros'  #

def explore_hdf5(filename):
    def print_structure(name, obj):
        print(f"{name} ({type(obj)})")
        if isinstance(obj, h5py.Dataset):
            print(f"  Shape: {obj.shape}")
            print(f"  Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  Contains: {list(obj.keys())}")

    with h5py.File(filename, 'r') as f:
        print("📁 HDF5 File Structure:")
        f.visititems(print_structure)

def subpixel_maximum(image2d):
    y, x = np.unravel_index(np.argmax(image2d), image2d.shape)

    if x < 1 or x > image2d.shape[1] - 2 or y < 1 or y > image2d.shape[0] - 2:
        return (y, x)  # fallback to integer location

    patch = image2d[y-1:y+2, x-1:x+2]
    X = np.array([
        [i**2, j**2, i*j, i, j, 1]
        for j in range(-1, 2)
        for i in range(-1, 2)
    ])
    Y = patch.flatten()
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    a, b, c, d, e, _ = coeffs
    A = np.array([[2*a, c], [c, 2*b]])
    b_vec = np.array([-d, -e])
    offset = np.linalg.solve(A, b_vec)
    return (y + offset[1], x + offset[0])

def subpixel_max_over_axes01(arr4d):
    H, W, D1, D2 = arr4d.shape
    result = np.zeros((D1, D2, 2))  # store (y, x) subpixel positions

    for i in range(D1):
        for j in range(D2):
            slice_2d = arr4d[:, :, i, j]
            result[i, j] = subpixel_maximum(slice_2d)

    return result

def build_design(u, v, order):
    terms = []
    if order >= 1:
        terms += [u, v]
    if order >= 2:
        terms += [u**2, u*v, v**2]
    if order >= 3:
        terms += [u**3, (u**2)*v, u*(v**2), v**3]
    return torch.stack(terms, dim=1)

# Fit function
def fit_and_residuals(order):
    design = build_design(u_flat, v_flat, order)
    design_w = design * W
    sol = torch.linalg.lstsq(design_w, DXY_w)
    M_fit = sol.solution
    pred = design @ M_fit
    return (DXY - pred).reshape(u_dim, v_dim, 2)

def build_design(u, v, order):
    terms = []
    if order >= 1:
        terms += [u, v]
    if order >= 2:
        terms += [u**2, u*v, v**2]
    if order >= 3:
        terms += [u**3, (u**2)*v, u*(v**2), v**3]
    if order >= 4:
        terms += [u**4, (u**3)*v, (u**2)*(v**2), u*(v**3), v**4]
    if order >= 5:
        terms += [u**5, (u**4)*v, (u**3)*(v**2), (u**2)*(v**3), u*(v**4), v**5]
    return torch.stack(terms, dim=1)

# --- Fit and get residuals ---
def fit_and_residuals(order):
    design = build_design(u_flat, v_flat, order)
    design_w = design * W
    sol = torch.linalg.lstsq(design_w, DXY_w)
    M_fit = sol.solution
    pred = design @ M_fit
    res = (DXY - pred).reshape(u_dim, v_dim, 2)
    rmse = torch.sqrt(torch.mean(res[...,0]**2 + res[...,1]**2))
    return res, rmse

# --- Magnitudes for coloring (with clipping) ---
def mag_clip(u, v):
    mag = torch.sqrt(u**2 + v**2)
    return torch.clamp(mag, max=clip_thresh)

def to_native_contig(a):
    if isinstance(a, np.ndarray):
        if a.dtype.byteorder not in ('=', '|'):
            a = a.byteswap().newbyteorder()
        if not a.flags['C_CONTIGUOUS']:
            a = np.ascontiguousarray(a)
        return a
    elif torch.is_tensor(a):
        return a
    else:
        raise TypeError("Expected NumPy array or Torch tensor")

def ensure_torch(a, device=DEVICE, dtype=DTYPE):
    if isinstance(a, np.ndarray):
        a = to_native_contig(a)
        return torch.from_numpy(a).to(dtype=dtype, device=device)
    elif torch.is_tensor(a):
        return a.to(dtype=dtype, device=device)
    else:
        raise TypeError("Expected NumPy array or Torch tensor")

def coerce_uv_map(arr, u_dim, v_dim, name="shift"):
    t = ensure_torch(arr)
    if t.shape == (u_dim, v_dim):
        return t
    if t.ndim == 1 and t.numel() == u_dim * v_dim:
        return t.view(u_dim, v_dim)
    raise ValueError(f"{name} must be shape [u_dim, v_dim] or flat length u_dim*v_dim; got {tuple(t.shape)}")

def align_and_upscale(F_xyuv, Sx_uv, Sy_uv, upscale=1, reduce='sum'):
    """
    Align F(x,y,u,v) using per-(u,v) shifts Sx(u,v), Sy(u,v), then aggregate over (u,v).
    Inputs:
      - F_xyuv: [x, y, u, v]
      - Sx_uv, Sy_uv: [u, v] or flat length u*v, measured shifts in +x/+y pixels
    Output:
      - aligned 2D image [y_out, x_out] after warping and sum/mean over (u,v)
    """
    # Enforce device/dtype
    F_xyuv = ensure_torch(F_xyuv)
    x_dim, y_dim, u_dim, v_dim = F_xyuv.shape  # NOTE: exact order per your spec

    Sx = coerce_uv_map(Sx_uv, u_dim, v_dim, name="Sx")
    Sy = coerce_uv_map(Sy_uv, u_dim, v_dim, name="Sy")

    # Prepare image batch for grid_sample: [N=u*v, C=1, H=y, W=x]
    # Permute [x,y,u,v] -> [u,v,y,x], then reshape
    F_uvyx = F_xyuv.permute(2, 3, 1, 0).contiguous()
    imgs = F_uvyx.reshape(u_dim * v_dim, 1, y_dim, x_dim)

    # Output resolution
    y_out = y_dim * upscale
    x_out = x_dim * upscale

    # Build target grid in pixel coords (x: 0..x_dim-1, y: 0..y_dim-1)
    y_coords_out = torch.linspace(0, y_dim - 1, y_out, device=imgs.device, dtype=imgs.dtype)
    x_coords_out = torch.linspace(0, x_dim - 1, x_out, device=imgs.device, dtype=imgs.dtype)
    Y_out, X_out = torch.meshgrid(y_coords_out, x_coords_out, indexing='ij')  # [H_out,W_out]

    # Expand to [u,v,H_out,W_out]
    X_exp = X_out.unsqueeze(0).unsqueeze(0).expand(u_dim, v_dim, y_out, x_out)
    Y_exp = Y_out.unsqueeze(0).unsqueeze(0).expand(u_dim, v_dim, y_out, x_out)

    # Broadcast per-(u,v) shifts and reverse them to align to reference
    dx = Sx[:, :, None, None]
    dy = Sy[:, :, None, None]
    X_src = X_exp - dx
    Y_src = Y_exp - dy

    # Normalize to [-1,1] for grid_sample; grid[...,0]=x, grid[...,1]=y
    if ALIGN_CORNERS:
        Xn = 2.0 * (X_src / (x_dim - 1)) - 1.0
        Yn = 2.0 * (Y_src / (y_dim - 1)) - 1.0
    else:
        Xn = (2.0 * (X_src + 0.5) / x_dim) - 1.0
        Yn = (2.0 * (Y_src + 0.5) / y_dim) - 1.0

    grid = torch.stack((Xn, Yn), dim=-1)  # [u,v,H_out,W_out,2]
    grid = grid.view(u_dim * v_dim, y_out, x_out, 2)

    # Warp
    warped = F.grid_sample(imgs, grid, mode='bicubic',
                           padding_mode=PADDING_MODE, align_corners=ALIGN_CORNERS) # bilinear
    # Aggregate over (u,v)
    if reduce == 'mean':
        aligned = warped.mean(dim=0).squeeze(0)  # [H_out,W_out]
    else:
        aligned = warped.sum(dim=0).squeeze(0)

    return aligned  # [y_out, x_out]

def ensure_torch(a, device=None, dtype=torch.float32):
    """
    Convert numpy.ndarray or torch.Tensor into a C-contiguous torch.Tensor
    on the given device & dtype.
    """
    if torch.is_tensor(a):
        return a.to(device or a.device, dtype=dtype).contiguous()
    # assume numpy array
    arr = a
    if arr.dtype.byteorder not in ('=', '|'):
        arr = arr.byteswap().newbyteorder()
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr).to(device or 'cpu', dtype=dtype)

def coerce_uv_map(uv, u_dim, v_dim, name="uv"):
    """
    Accepts uv as shape [u,v] or flat [u*v], returns a [u,v] torch.Tensor.
    """
    t = ensure_torch(uv)
    if t.ndim == 2 and tuple(t.shape) == (u_dim, v_dim):
        return t
    if t.ndim == 1 and t.numel() == u_dim * v_dim:
        return t.view(u_dim, v_dim)
    raise ValueError(f"{name} must be [u_dim,v_dim] or flat length {u_dim*v_dim}; got {tuple(t.shape)}")

def align_and_upscale_batch_grid(
    F_xyuv,         # [x, y, u, v], numpy or Torch
    Sx_uv,          # [u, v] or flat u*v, numpy or Torch
    Sy_uv,          # [u, v] or flat u*v, numpy or Torch
    upscale=1,      # integer scale factor
    chunk_size=256, # how many views per batch
    reduce='mean'   # 'sum' or 'mean' over (u,v)
):
    # 1) Convert inputs
    F_xyuv = ensure_torch(F_xyuv)
    x_dim, y_dim, u_dim, v_dim = F_xyuv.shape

    Sx = coerce_uv_map(Sx_uv, u_dim, v_dim, name="Sx")
    Sy = coerce_uv_map(Sy_uv, u_dim, v_dim, name="Sy")

    # 2) Permute to [u, v, y, x] and batchify → [B,1,y,x]
    B = u_dim * v_dim
    F_batch = F_xyuv.permute(2, 3, 1, 0).reshape(B, 1, y_dim, x_dim)

    # 3) Flatten shifts → [B,2]
    shifts = torch.stack([Sx.flatten(), Sy.flatten()], dim=1).to(F_batch)

    # 4) Prepare output resolution + base grid
    Y_out = y_dim * upscale
    X_out = x_dim * upscale
    xs = torch.linspace(-1, 1, X_out, device=F_batch.device)
    ys = torch.linspace(-1, 1, Y_out, device=F_batch.device)
    base_grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1)
    # base_grid: [Y_out, X_out, 2]

    # 5) Accumulators
    out = torch.zeros((Y_out, X_out),
                      device=F_batch.device,
                      dtype=F_batch.dtype)
    cnt = torch.zeros_like(out)

    # 6) Process in chunks of views
    for start in range(0, B, chunk_size):
        end  = min(start + chunk_size, B)
        imgs = F_batch[start:end]    # [C,1,y,x]
        sb   = shifts[start:end]     # [C,2]

        # normalize shifts to grid units (undo measured +Sx → sample from x−Sx)
        dxn = sb[:, 0:1].unsqueeze(-1).unsqueeze(-1) * (2.0 / (x_dim - 1))
        dyn = sb[:, 1:2].unsqueeze(-1).unsqueeze(-1) * (2.0 / (y_dim - 1))
        # build per-image grids [C, Y_out, X_out, 2]
        grid_b = base_grid.unsqueeze(0) + \
                 torch.cat([dxn, dyn], dim=1).permute(0, 2, 3, 1)

        # nearest‐neighbor sampling
        sampled = F.grid_sample(
            imgs, grid_b,
            mode='nearest',
            padding_mode=PADDING_MODE,
            align_corners=ALIGN_CORNERS
        ).squeeze(1)  # → [C, Y_out, X_out]

        out += sampled.sum(dim=0)
        cnt += (sampled != 0).to(out.dtype).sum(dim=0)

    # 7) Final reduction
    if reduce == 'mean':
        safe = torch.where(cnt > 0, cnt, torch.ones_like(cnt))
        out  = out / safe

    return out.cpu().numpy()


# --- Helper: build polynomial design matrix up to 5th order ---
def build_design(u, v, order):
    # constant term + up to nth-order u,v powers
    terms = [torch.ones_like(u)]
    if order >= 1:
        terms += [u, v]
    if order >= 2:
        terms += [u**2, u*v, v**2]
    if order >= 3:
        terms += [u**3, (u**2)*v, u*(v**2), v**3]
    if order >= 4:
        terms += [u**4, (u**3)*v, (u**2)*(v**2), u*(v**3), v**4]
    if order >= 5:
        terms += [u**5, (u**4)*v, (u**3)*(v**2), (u**2)*(v**3), u*(v**4), v**5]
    return torch.stack(terms, dim=1)


# --- Fit, predict, and compute residuals ---
def fit_predict_and_residuals(order, xm_sub, ym_sub, W):
    """
    Returns:
      res_xy    - [u_dim, v_dim, 2] residuals (meas – pred)
      pred_xy   - [u_dim, v_dim, 2] predicted displacements
      rmse      - scalar root‐mean‐square error
      M_fit     - [n_terms, 2] polynomial coefficients for (dx,dy)
    """
    u_dim, v_dim = xm_sub.shape
    cu, cv = u_dim // 2, v_dim // 2

    # measured offsets about center
    dx_map = xm_sub - xm_sub[cu, cv]
    dy_map = ym_sub - ym_sub[cu, cv]

    # flatten coords, data, weights
    uu, vv = torch.meshgrid(
        torch.arange(u_dim, dtype=torch.float32),
        torch.arange(v_dim, dtype=torch.float32),
        indexing='ij'
    )
    u_rel = (uu - cu).reshape(-1)
    v_rel = (vv - cv).reshape(-1)
    dx_flat = dx_map.reshape(-1)
    dy_flat = dy_map.reshape(-1)
    W_flat  = W.reshape(-1)

    # build and weight design + data
    A     = build_design(u_rel, v_rel, order)       # [N, n_terms]
    DXY   = torch.stack([dx_flat, dy_flat], dim=1)  # [N, 2]
    w_s   = torch.sqrt(W_flat.clamp(min=0) + 1e-12)
    A_w   = A * w_s.unsqueeze(1)
    DXY_w = DXY * w_s.unsqueeze(1)

    # solve weighted LS
    sol     = torch.linalg.lstsq(A_w, DXY_w)
    M_fit   = sol.solution                          # [n_terms,2]

    # predict & reshape
    pred     = A @ M_fit                            # [N,2]
    pred_xy  = pred.reshape(u_dim, v_dim, 2)        # [u,v,2]
    res_xy   = (DXY.reshape(u_dim, v_dim, 2) - pred_xy)

    # compute overall RMSE
    rmse = torch.sqrt((res_xy[...,0]**2 + res_xy[...,1]**2).mean()).item()

    return res_xy, pred_xy, rmse, M_fit


def compute_gaussian_weights(u_dim, v_dim, sigma_frac=0.3):
    """
    Returns a [u_dim, v_dim] tensor of Gaussian weights peaked at the center.
    
    sigma_frac: controls spread; 
      smaller → sharper peak (more focus on center),
      recommended between 0.2 and 0.5.
    """
    # create coordinate grids
    uu, vv = torch.meshgrid(
        torch.arange(u_dim, dtype=torch.float32),
        torch.arange(v_dim, dtype=torch.float32),
        indexing='ij'
    )
    # center indices
    cu, cv = (u_dim - 1) / 2, (v_dim - 1) / 2

    # standard deviations
    sigma_u = sigma_frac * u_dim
    sigma_v = sigma_frac * v_dim

    gauss_u = torch.exp(-0.5 * ((uu - cu) / sigma_u)**2)
    gauss_v = torch.exp(-0.5 * ((vv - cv) / sigma_v)**2)

    # outer product gives 2D Gaussian
    W = gauss_u * gauss_v
    return W

def ensure_torch(a, device='cpu', dtype=torch.float32):
    """
    Convert numpy or torch → dense, C-contiguous FloatTensor on device.
    Densify sparse_coo tensors automatically.
    """
    if torch.is_tensor(a):
        t = a.to(device=device, dtype=dtype)
        if t.is_sparse:
            t = t.to_dense()
        return t.contiguous()
    arr = np.asarray(a)
    if arr.dtype.byteorder not in ('=', '|'):
        arr = arr.byteswap().newbyteorder()
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)

def coerce_uv_map(uv, U, V, name):
    """
    Ensures uv is [U,V], accepting flat length U*V.
    """
    t = ensure_torch(uv)
    if t.shape == (U, V):
        return t
    if t.ndim == 1 and t.numel() == U*V:
        return t.view(U, V)
    raise ValueError(f"{name} must be [U,V] or flat length U*V; got {tuple(t.shape)}")

# 2) Weight window (Hann)
# -------------------------------------------------------------------
def compute_hann_weights(U, V):
    """
    Returns a [U,V] Hann window: 1 at center, 0 at edges.
    """
    wu = torch.hann_window(U, periodic=False, dtype=torch.float32)
    wv = torch.hann_window(V, periodic=False, dtype=torch.float32)
    return wu.unsqueeze(1) * wv.unsqueeze(0)

# 3) Polynomial design matrix
# -------------------------------------------------------------------
def build_design(u, v, order):
    """
    u,v: 1D tensors of length N = U*V, giving coords relative to center.
    Returns [N, n_terms], with a constant term first.
    """
    terms = [torch.ones_like(u)]
    if order >= 1:
        terms += [u, v]
    if order >= 2:
        terms += [u**2, u*v, v**2]
    if order >= 3:
        terms += [u**3, (u**2)*v, u*(v**2), v**3]
    if order >= 4:
        terms += [u**4, (u**3)*v, (u**2)*(v**2), u*(v**3), v**4]
    if order >= 5:
        terms += [u**5, (u**4)*v, (u**3)*(v**2), (u**2)*(v**3), u*(v**4), v**5]
    return torch.stack(terms, dim=1)

# 4) Fit, predict, residuals
# -------------------------------------------------------------------
def align_and_upscale_batch_grid(
    F_xyuv,    # [X,Y,U,V] or 3D [X,Y,U]; numpy or torch (dense/sparse)
    Sx_uv,     # [U,V]
    Sy_uv,     # [U,V]
    upscale=1,
    chunk_size=256,
    reduce='mean'
):
    # 1) to torch & densify
    F_t = ensure_torch(F_xyuv)
    # 2) if 3D, assume V=1
    if F_t.dim() == 3:
        F_t = F_t.unsqueeze(-1)
    if F_t.dim() != 4:
        raise ValueError(f"F_xyuv must be 4D [X,Y,U,V]; got shape {tuple(F_t.shape)}")
    X, Y, U, V = F_t.shape

    # 3) coerce shifts to [U,V]
    def coerce(uv, name):
        t = ensure_torch(uv, device=F_t.device)
        if t.shape == (U,V):
            return t
        if t.ndim==1 and t.numel()==U*V:
            return t.view(U,V)
        raise ValueError(f"{name} must be [U,V] or flat length U*V; got {tuple(t.shape)}")
    Sx = coerce(Sx_uv, 'Sx_uv')
    Sy = coerce(Sy_uv, 'Sy_uv')

    # 4) batchify to [B,1,Y,X]
    B = U*V
    batch = F_t.permute(2,3,1,0).reshape(B,1,Y,X)

    # 5) flatten shifts → [B,2]
    shifts = torch.stack([Sx.flatten(), Sy.flatten()], dim=1).to(batch.device)

    # 6) build normalized grid once
    Y2, X2 = Y*upscale, X*upscale
    xs = torch.linspace(-1, 1, X2, device=batch.device)
    ys = torch.linspace(-1, 1, Y2, device=batch.device)
    base = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1)  # [Y2,X2,2]

    out = torch.zeros(Y2, X2, device=batch.device)
    cnt = torch.zeros_like(out)

    # 7) chunked sampling
    for i in range(0, B, chunk_size):
        j = min(i+chunk_size, B)
        imgs = batch[i:j]         # [C,1,Y,X]
        sb   = shifts[i:j]        # [C,2]

        # convert shifts into normalized grid offsets
        dxn = sb[:,0:1].view(-1,1,1) * (2.0/(X-1))
        dyn = sb[:,1:2].view(-1,1,1) * (2.0/(Y-1))

        grid = base.unsqueeze(0) \
             + torch.cat([dxn,dyn],dim=1).permute(0,2,3,1)

        samp = F.grid_sample(
            imgs, grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(1)

        out += samp.sum(0)
        cnt += (samp!=0).sum(0).to(out.dtype)

    if reduce=='mean':
        safe = torch.where(cnt>0, cnt, torch.ones_like(cnt))
        out = out/safe

    return out.cpu().numpy()

# 5) Nearest-neighbor batch alignment & upscaling
# -------------------------------------------------------------------
def align_and_upscale_batch_grid(
    F_xyuv, Sx_uv, Sy_uv, upscale=1, chunk_size=256, reduce='mean'
):
    F_t = ensure_torch(F_xyuv)
    if F_t.dim() == 3:
        F_t = F_t.unsqueeze(-1)
    if F_t.dim() != 4:
        raise ValueError(f"Expected 4D [X,Y,U,V], got {tuple(F_t.shape)}")
    X,Y,U,V = F_t.shape

    # shifts → dense [U,V]
    def coerce(uvar, name):
        t = ensure_torch(uvar, device=F_t.device)
        if t.shape == (U,V):
            return t
        if t.ndim==1 and t.numel()==U*V:
            return t.view(U,V)
        raise ValueError(f"{name} must be [U,V] or flat length U*V")
    Sx = coerce(Sx_uv, 'Sx_uv')
    Sy = coerce(Sy_uv, 'Sy_uv')

    # batchify [U,V,Y,X]
    B = U*V
    batch = F_t.permute(2,3,1,0).reshape(B,1,Y,X)
    shifts = torch.stack([Sx.flatten(), Sy.flatten()], dim=1).to(batch.device)

    # precompute base grid [Y2,X2,2]
    Y2, X2 = Y*upscale, X*upscale
    xs = torch.linspace(-1,1,X2,device=batch.device)
    ys = torch.linspace(-1,1,Y2,device=batch.device)
    base = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1)

    out = torch.zeros(Y2,X2,device=batch.device)
    cnt = torch.zeros_like(out)

    for i in range(0, B, chunk_size):
        j = min(i+chunk_size, B)
        imgs = batch[i:j]       # [C,1,Y,X]
        sb   = shifts[i:j]      # [C,2]

        # build a [C,1,1,2] offset tensor
        dxn = sb[:,0].view(-1,1,1,1) * (2.0/(X-1))
        dyn = sb[:,1].view(-1,1,1,1) * (2.0/(Y-1))
        offsets = torch.cat([dxn, dyn], dim=3)  # [C,1,1,2]

        # broadcast-add to get [C,Y2,X2,2]
        grid = base.unsqueeze(0) + offsets

        samp = F.grid_sample(
            imgs, grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(1)

        out += samp.sum(0)
        cnt += (samp!=0).sum(0).to(out.dtype)

    if reduce=='mean':
        safe = torch.where(cnt>0, cnt, torch.ones_like(cnt))
        out = out/safe

    return out.cpu().numpy()
