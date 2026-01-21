from time import perf_counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import RayTraceTorch as rtt


def test_rendering_pipeline():
    print("--- Starting Headless Render Test ---")

    # A. Setup Scene
    scene = rtt.scene.Scene()

    # Create a dummy element
    element1 = rtt.elements.Element()
    element1.shape = rtt.geom.Sphere(radius=5.0)
    element1.surface_functions.append(rtt.phys.Reflect())

    element2 = rtt.elements.Element()
    element2.shape = rtt.geom.Sphere(radius=5.0, transform=rtt.geom.RayTransform(translation = [0, -3, 10]))
    element2.surface_functions.append(rtt.phys.Reflect())

    shape3 = rtt.geom.Box(height=7, width=7, length=7)
    element3 = rtt.elements.Element()
    element3.shape = shape3
    for _ in range(len(element3.shape)):
        element3.surface_functions.append(rtt.phys.Block())

    scene.elements.append(element1)
    scene.elements.append(element2)
    scene.elements.append(element3)

    scene._build_index_maps()
    print("Scene setup complete.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # B. Setup Camera
    # Positioned at Z=-20, looking at Z=0
    cam = rtt.render.Camera(
        position=(-30, 0, 0),
        look_at=(0, 0, 0),
        up_vector=(0, 1, 0),
        fov_deg=40,
        width=1024,
        height=1024,
        device=device # Use CPU for safety in test
    )
    
    print("Camera setup complete.")

    scene.to('cuda')
    # C. Initialize Renderer
    # Background color: Dark Blue (0.1, 0.1, 0.3)
    renderer = rtt.render.Renderer(scene, background_color=(0.1, 0.1, 0.1))

    # D. Execute Render (The Critical Step)
    try:
        print("Rendering...")
        t0 = perf_counter()
        with torch.no_grad():
            image_tensor = renderer.render_3d(cam)
        tf = perf_counter() - t0

        print(f"Render successful. Output shape: {image_tensor.shape}")
        print(f"Rendering took {tf:.3f} seconds. ({1/tf:.1f}fps)")

        # Validation
        if torch.isnan(image_tensor).any():
            print("ERROR: Render contains NaNs!")
        if image_tensor.max() > 1.0 or image_tensor.min() < 0.0:
            print("WARNING: Render values out of range [0,1]")

    except Exception as e:
        print(f"CRITICAL FAILURE during render_3d: {e}")
        import traceback
        traceback.print_exc()
        return

    # E. Save Output
    try:
        save_path = "test_render_output.png"
        plt.imsave(save_path, image_tensor.numpy())
        print(f"Image saved to {os.path.abspath(save_path)}")

        # Check if image is just background (all pixels same)
        is_flat = torch.all(image_tensor == image_tensor[0, 0])
        if is_flat:
            print("WARNING: Image is completely flat color. Rays might have missed.")
        else:
            print("SUCCESS: Image contains variation (geometry detected).")

    except Exception as e:
        print(f"Error saving image: {e}")


if __name__ == "__main__":
    test_rendering_pipeline()