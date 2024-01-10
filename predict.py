# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
from cog import BasePredictor, Input, Path



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Change directory and run the installation commands
        try:
            os.chdir("MeshGraphormer")
            subprocess.run(["pip", "install", "-e", "."], check=True)
            subprocess.run(["pip", "install", "-e", "./manopth/."], check=True)
            os.chdir("..")  # Change back to the original directory

            # # Search for the file
            # for root, dirs, files in os.walk("/src"):
            #     if "MANO_RIGHT.pkl" in files:
            #         src = os.path.join(root, "MANO_RIGHT.pkl")
            #         print(f"Found file at {src}")
            #         dst = "/src/modeling/data/MANO_RIGHT.pkl"
                    
            #         # Create directories for the destination path if they do not exist
            #         os.makedirs(os.path.dirname(dst), exist_ok=True)
                    
            #         # Create symbolic link
            #         os.symlink(src, dst)
            #         print(f"Created symbolic link from {src} to {dst}")

        except subprocess.CalledProcessError as e:
            print("[!] An error occurred while installing MeshGraphormer packages.")
            raise e

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt describing the desired hand gesture"),
        strength: float = Input(description="Strength of the effect", ge=0.0, le=1.0, default=0.55),
        seed: int = Input(description="Random seed for generation", ge=0, le=1000000, default=None)
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        command = f"python handrefiner.py --input_img {str(image)} --out_dir output --strength {strength} --weights models/inpaint_depth_control.ckpt --prompt '{prompt}' --seed {seed}"
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output_path = 'output/image_0.jpg'
            return output_path
        except subprocess.CalledProcessError as e:
            print("Error:", e.stderr)
            return None


        
