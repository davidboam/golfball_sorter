import torch
import torch.nn.functional as F
import numpy as np
import cv2

def _normalize_heatmap(cam):
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam

def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    out = np.clip((1 - alpha) * img_rgb + alpha * heatmap, 0, 255).astype(np.uint8)
    return out

def gradcam_for_view(model, img_tensor, aux_tensor, target_head="grade", class_idx=None, n_views=4, device="cpu"):
    model.eval()
    img_tensor = img_tensor.to(device)
    aux_tensor = aux_tensor.to(device)

    feats_spatial = {}
    grads = {}

    def fwd_hook(m, inp, out):
        feats_spatial['x'] = out

    def bwd_hook(m, gin, gout):
        grads['x'] = gout[0]

    handle_f = model.encoder.forward_features.register_forward_hook(fwd_hook)
    handle_b = model.encoder.forward_features.register_full_backward_hook(bwd_hook)

    # replicate this image across views
    views = img_tensor.unsqueeze(1).repeat(1, n_views, 1, 1, 1)

    out = model(views, aux_tensor)

    if target_head == "brand":
        blog = out["brand_logits"]
        probs = F.softmax(blog, dim=1)
        c = probs.argmax(dim=1).item() if class_idx is None else int(class_idx)
        target_logit = blog[0, c]
    else:
        from ballnet.models.heads import ordinal_logits_to_probs
        glog = out["grade_logits"]
        probs = ordinal_logits_to_probs(glog)
        c = probs.argmax(dim=1).item() if class_idx is None else int(class_idx)
        target_logit = glog[0, :c].sum() if c > 0 else (glog[0, 0] * 0.0)

    model.zero_grad(set_to_none=True)
    target_logit.backward(retain_graph=True)

    fmap = feats_spatial['x'].detach()
    grad = grads['x'].detach()
    weights = grad.mean(dim=(2,3), keepdim=True)
    cam = (weights * fmap).sum(dim=1, keepdim=False)
    cam = F.relu(cam)[0].cpu().numpy()
    cam = _normalize_heatmap(cam)

    H = img_tensor.shape[2]; W = img_tensor.shape[3]
    cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    handle_f.remove(); handle_b.remove()
    return cam
