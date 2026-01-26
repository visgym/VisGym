import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Launch scripts locally or in Docker.")
    parser.add_argument("--run_script_path", type=str, required=True, help="Path to the script to run.")
    parser.add_argument("--docker", action="store_true", help="Run inside a Docker container.")
    parser.add_argument("--image", type=str, default="hiyouga/llamafactory:latest", help="Docker image to use.")
    parser.add_argument("--gpus", type=str, default="all", help="GPUs to use (e.g., 'all', '0', '0,1').")
    
    args, unknown = parser.parse_known_args()
    
    script_path = args.run_script_path
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    # Make script executable
    os.chmod(script_path, 0o755)

    if args.docker:
        # Check if we should build the image
        # For simplicity, let's assume if a Dockerfile exists and the user didn't specify a custom image (or specified 'local'), we build.
        if args.image == "local" or (args.image == "hiyouga/llamafactory:latest" and os.path.exists("Dockerfile")):
            print("Building Docker image from Dockerfile...")
            image_name = "launch_gym_local"
            build_cmd = ["docker", "build", "-t", image_name, "."]
            try:
                subprocess.run(build_cmd, check=True)
                args.image = image_name
            except subprocess.CalledProcessError as e:
                print(f"Failed to build Docker image: {e}")
                sys.exit(1)

        print(f"Launching {script_path} in Docker container ({args.image})...")
        
        # Get absolute path for mounting
        cwd = os.getcwd()
        
        # Determine data directory on host
        # Check ../../data/gym (standard structure) then ./data/gym (local download)
        host_data_dir = os.path.join(cwd, "../../data/gym")
        if not os.path.exists(host_data_dir):
            host_data_dir = os.path.join(cwd, "data/gym")
        
        # Construct docker command
        docker_cmd = [
            "docker", "run", "--rm", "-it",
            "--gpus", args.gpus,
            "--ipc=host", # Recommended for PyTorch
            "-v", f"{cwd}:/home/clouduser/Code/Github/visgym_training", # Mount to expected path
            "-v", f"{host_data_dir}:/home/clouduser/Code/data/gym", # Mount data
            "-w", "/home/clouduser/Code/Github/visgym_training",
            args.image,
            "bash", script_path
        ]
        
        # Add any extra arguments passed to the script
        if unknown:
            docker_cmd.extend(unknown)
            
        print("Running command:", " ".join(docker_cmd))
        try:
            subprocess.run(docker_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Docker execution failed with exit code {e.returncode}")
            sys.exit(e.returncode)
            
    else:
        print(f"Launching {script_path} locally...")
        
        # Read the script content
        with open(script_path, 'r') as f:
            script_content = f.read()
            
        # Define replacements
        # We assume the script is run from the repo root, so os.getcwd() is the launch dir
        repo_root = os.getcwd()
        
        # Default local paths (can be improved with arguments)
        local_data_dir = os.path.join(repo_root, "data", "gym")
        local_models_dir = os.path.join(repo_root, "models") # Assumption
        
        # Perform replacements
        # 1. Repo root
        script_content = script_content.replace("/home/clouduser/Code/Github/visgym_training", repo_root)
        
        # 2. Data directory
        # If the user downloaded data to ./data/gym using data_transfer.sh locally
        script_content = script_content.replace("/home/clouduser/Code/data/gym", local_data_dir)
        
        # 3. Models directory
        # This is tricky as we don't know where the user keeps models locally.
        # We'll replace it with a local 'models' dir and warn the user.
        script_content = script_content.replace("/home/clouduser/Code/Models", local_models_dir)
        
        # Write to a temporary file
        import tempfile
        import stat
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_script:
            temp_script.write(script_content)
            temp_script_path = temp_script.name
            
        # Make executable
        st = os.stat(temp_script_path)
        os.chmod(temp_script_path, st.st_mode | stat.S_IEXEC)
        
        print(f"Created temporary script with local paths: {temp_script_path}")
        print(f"  Replaced '/home/clouduser/Code/Github/visgym_training' with '{repo_root}'")
        print(f"  Replaced '/home/clouduser/Code/data/gym' with '{local_data_dir}'")
        print(f"  Replaced '/home/clouduser/Code/Models' with '{local_models_dir}'")
        
        cmd = ["bash", temp_script_path]
        if unknown:
            cmd.extend(unknown)
            
        print("Running command:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Execution failed with exit code {e.returncode}")
            # Don't delete temp file on failure for debugging
            print(f"Temporary script preserved at {temp_script_path}")
            sys.exit(e.returncode)
        finally:
             # Cleanup if successful (or maybe just leave it? No, better clean up if successful)
             if os.path.exists(temp_script_path):
                 try:
                     os.remove(temp_script_path)
                 except:
                     pass

if __name__ == "__main__":
    main()
