def requirements_to_python_env(requirements_path="requirements.txt", output_path="python_env.yaml", python_version=None):
    with open(requirements_path, "r") as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    with open(output_path, "w") as f:
        f.write("name: myenv\n")
        f.write("dependencies:\n")
        if python_version:
            f.write(f"  - python={python_version}\n")
        f.write("  - pip:\n")
        for pkg in packages:
            f.write(f"    - {pkg}\n")

    print(f"Created {output_path} from {requirements_path}")
    
if __name__ == "__main__":
    requirements_to_python_env(python_version="3.11.13")

# Example usage:
# requirements_to_python_env(python_version="3.8")

