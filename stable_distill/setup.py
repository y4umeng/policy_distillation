import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="rl-distiller",
        version="0.1.0",
        description="A KD framework for RL based on stable-baselines3",
        packages=setuptools.find_packages(),
        python_requires=">=3.7",
        install_requires=["stable-baselines3>=1.7.0", "gym", "torch>=1.9", "pyyaml"],
    )