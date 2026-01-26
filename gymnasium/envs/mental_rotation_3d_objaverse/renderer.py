import os, numpy as np, pyrender
import getpass
from PIL import Image
import trimesh as tm
import gc
import subprocess
# check number of gpus and randomly select one
try:
    # Use nvidia-smi to get GPU count
    result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                          capture_output=True, text=True, check=True)
    num_gpus = len(result.stdout.strip().split('\n'))
except (subprocess.CalledProcessError, FileNotFoundError):
    # Fallback: assume 1 GPU if nvidia-smi is not available
    num_gpus = 1

best_gpu_local = np.random.randint(0, num_gpus) if num_gpus > 0 else 0

# Get the actual GPU ID from CUDA_VISIBLE_DEVICES if set
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if cuda_visible_devices:
    visible_gpus = [int(x.strip()) for x in cuda_visible_devices.split(',')]
    best_gpu_global = visible_gpus[best_gpu_local]
    print(f"Using GPU {best_gpu_global} (local index {best_gpu_local} from CUDA_VISIBLE_DEVICES={cuda_visible_devices})")
    os.environ['EGL_DEVICE_ID'] = str(best_gpu_global)
else:
    print(f"Using GPU {best_gpu_local}")
    os.environ['EGL_DEVICE_ID'] = str(best_gpu_local)

user = os.environ.get("USER") or getpass.getuser() or "default"
xdg_dir = f"/tmp/runtime-{user}"
os.makedirs(xdg_dir, exist_ok=True)
os.environ["XDG_RUNTIME_DIR"] = xdg_dir
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "egl"
os.environ["DISPLAY"] = ""
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu/nvidia:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LIBGL_DRIVERS_PATH"] = "/usr/lib/x86_64-linux-gnu/nvidia"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["FILAMENT_BACKEND"] = "egl"
os.environ["OPEN3D_LOG_LEVEL"] = "Error"

def _fxfy_from_fov_res(fov, W, H):
    f = np.asarray(fov, dtype=float).ravel()
    if f.size == 1:
        fy = H/(2*np.tan(np.deg2rad(f[0]/2))); fx = fy*W/H
    else:
        fx = W/(2*np.tan(np.deg2rad(f[0]/2)))
        fy = H/(2*np.tan(np.deg2rad(f[1]/2)))
    return float(fx), float(fy)

class FastRenderer:
    def __init__(self, w=640, h=480):
        self.w, self.h = int(w), int(h)
        self.r = pyrender.OffscreenRenderer(self.w, self.h)

    def _filter_supported_geometry(self, sc_tm: tm.Scene) -> tm.Scene:
        """
        Build a new trimesh.Scene containing only triangle meshes supported by pyrender.
        - Keep Trimesh with faces > 0
        - Try to convert non-mesh geometries (e.g., Path3D) via to_mesh/to_trimesh/triangulate
        - Skip anything unsupported
        Preserve camera and camera_transform when available.
        """
        try:
            new_sc = tm.Scene()
            flat = sc_tm.graph.to_flattened()
            for node_name, info in flat.items():
                geom_name = info.get("geometry")
                if geom_name is None:
                    continue
                g = sc_tm.geometry.get(geom_name)
                if g is None:
                    continue
                mesh = None
                # If it's already a Trimesh with faces
                if isinstance(g, tm.Trimesh) and len(getattr(g, "faces", [])) > 0:
                    mesh = g.copy()
                else:
                    # Attempt conversions commonly available on Path3D or other types
                    for mname in ("to_mesh", "to_trimesh", "triangulate"):
                        if hasattr(g, mname):
                            try:
                                cand = getattr(g, mname)()
                                if isinstance(cand, tm.Trimesh) and len(getattr(cand, "faces", [])) > 0:
                                    mesh = cand
                                    break
                                if isinstance(cand, (list, tuple)):
                                    # Merge list of meshes if possible
                                    sub = [c for c in cand if isinstance(c, tm.Trimesh) and len(getattr(c, "faces", [])) > 0]
                                    if sub:
                                        try:
                                            mesh = tm.util.concatenate(sub)
                                            break
                                        except Exception:
                                            # If concatenate fails, just take first
                                            mesh = sub[0]
                                            break
                            except Exception:
                                continue
                if mesh is None:
                    # skip unsupported geometry (e.g., Path3D without conversion)
                    continue
                try:
                    new_sc.add_geometry(mesh, node_name=node_name, transform=info.get("transform", None))
                except Exception:
                    # If add fails, skip this geometry
                    continue

            # Preserve camera if present
            for attr in ("camera", "camera_transform"):
                if hasattr(sc_tm, attr):
                    try:
                        setattr(new_sc, attr, getattr(sc_tm, attr))
                    except Exception:
                        pass
            return new_sc
        except Exception as e:
            # If anything goes wrong, return original to try sanitization path
            print(f"Warning: Could not filter geometry from scene: {e}. Returning original scene.")
            return sc_tm

    def _sanitize_trimesh_textures(self, sc_tm):
        """
        Convert any numpy.ndarray textures to PIL.Image or strip them to avoid
        pyrender complaining about invalid texture types.
        Returns a shallow-copied trimesh.Scene with sanitized materials.
        """
        def _ensure_pil_rgba(tex):
            # Convert numpy arrays or PIL images to RGBA PIL.Image
            try:
                if isinstance(tex, Image.Image):
                    if tex.mode in ("RGB", "RGBA"):
                        return tex.convert("RGBA")
                    # Convert other modes (L, LA, P, I, etc.) to RGBA
                    return tex.convert("RGBA")
                import numpy as _np
                if isinstance(tex, _np.ndarray):
                    arr = tex
                    if arr.dtype != _np.uint8:
                        arr = _np.clip(arr, 0, 255).astype(_np.uint8)
                    if arr.ndim == 2:
                        # HxW -> replicate to RGB
                        arr = _np.stack([arr, arr, arr, _np.full_like(arr, 255)], axis=-1)
                    elif arr.ndim == 3:
                        h, w, c = arr.shape
                        if c == 1:
                            arr = _np.concatenate([arr, arr, arr, _np.full((h, w, 1), 255, dtype=_np.uint8)], axis=-1)
                        elif c == 2:
                            # Common for metallic-roughness packed textures; expand to RGBA (R,G,0,255)
                            zeros = _np.zeros((h, w, 1), dtype=_np.uint8)
                            alpha = _np.full((h, w, 1), 255, dtype=_np.uint8)
                            arr = _np.concatenate([arr, zeros, alpha], axis=-1)
                        elif c == 3:
                            alpha = _np.full((h, w, 1), 255, dtype=_np.uint8)
                            arr = _np.concatenate([arr, alpha], axis=-1)
                        elif c >= 4:
                            arr = arr[..., :4]
                    return Image.fromarray(arr, mode="RGBA")
            except Exception:
                return None
            return None

        try:
            sc_copy = sc_tm.copy()
        except Exception:
            sc_copy = sc_tm

        try:
            for gname, geom in sc_copy.geometry.items():
                vis = getattr(geom, "visual", None)
                mat = getattr(vis, "material", None)
                if mat is None:
                    continue
                # Common attributes seen in trimesh materials / glTF PBR
                for attr in (
                    "image",
                    "baseColorTexture",
                    "metallicRoughnessTexture",
                    "normalTexture",
                    "occlusionTexture",
                    "emissiveTexture",
                ):
                    if hasattr(mat, attr):
                        try:
                            tex = getattr(mat, attr)
                            if tex is None:
                                continue
                            # Coerce any texture to RGBA PIL or drop it
                            coerced = _ensure_pil_rgba(tex)
                            if coerced is not None:
                                setattr(mat, attr, coerced)
                            else:
                                # If cannot coerce, drop to avoid renderer failure
                                setattr(mat, attr, None)
                        except Exception:
                            # If we can't access the attribute, skip
                            pass
        except Exception:
            # Fail-open: if sanitization itself fails, just return original
            return sc_tm
        return sc_copy

    def _add_camera(self, sc_tm, sc_py):
        cam = getattr(sc_tm, "camera", None)
        znear = getattr(cam, "znear", 1e-3) if cam is not None else 1e-3
        zfar  = getattr(cam, "zfar", 1e4)  if cam is not None else 1e4

        if cam is not None and getattr(cam, "fov", None) is not None:
            fx, fy = _fxfy_from_fov_res(cam.fov, self.w, self.h)
            cx, cy = self.w / 2.0, self.h / 2.0
            pc = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=znear, zfar=zfar)
        else:
            pc = pyrender.PerspectiveCamera(
                yfov=np.deg2rad(45.0),
                aspectRatio=self.w / float(self.h),
                znear=znear,
                zfar=zfar
            )

        pose = getattr(sc_tm, "camera_transform", None)
        if pose is None:
            pose = np.eye(4)
        sc_py.add(pc, pose=pose)

    def render(self, tm_scene):
        # Handle None or invalid scene gracefully
        if tm_scene is None:
            print("Warning: tm_scene is None, returning blank white image.")
            return np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        
        if self.r is None:
            self.r = pyrender.OffscreenRenderer(self.w, self.h)
        
        # First, filter out unsupported geometry (e.g., Path3D) so pyrender only sees triangle meshes
        filtered_scene = self._filter_supported_geometry(tm_scene)
        
        # Check if filtering returned None or invalid scene
        if filtered_scene is None or not hasattr(filtered_scene, 'geometry'):
            print("Warning: filtered_scene is invalid, returning blank white image.")
            return np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        
        # Try to create the scene; if textures are numpy arrays or non-RGBA, sanitize them
        try:
            sc = pyrender.Scene.from_trimesh_scene(
                filtered_scene, ambient_light=[0.25, 0.25, 0.25], bg_color=[1.0, 1.0, 1.0, 1.0]
            )
        except Exception as e:
            # Attempt a sanitized copy (convert/drop numpy textures)
            try:
                tm_scene_sanitized = self._sanitize_trimesh_textures(filtered_scene)
                sc = pyrender.Scene.from_trimesh_scene(
                    tm_scene_sanitized, ambient_light=[0.25, 0.25, 0.25], bg_color=[1.0, 1.0, 1.0, 1.0]
                )
            except Exception as e2:
                print(f"Warning: Could not create pyrender scene: {e2}. Returning blank white image.")
                return np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        
        self._add_camera(tm_scene, sc)
        sc.add(pyrender.DirectionalLight(intensity=2.0), pose=np.eye(4))
        color, _ = self.r.render(sc)
        del sc
        return color
    
    def clear_cache(self):
        if self.r is not None:
            try:
                self.r.delete()
            except Exception as e:
                # EGL context may be invalid, just clear the reference
                print(f"Warning: Could not delete renderer: {e}")
            finally:
                self.r = None
        gc.collect()
        
    def close(self):
        if self.r is not None:
            try:
                self.r.delete()
            except Exception as e:
                # EGL context may be invalid, just clear the reference
                print(f"Warning: Could not delete renderer: {e}")
            finally:
                self.r = None

