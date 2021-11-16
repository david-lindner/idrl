import setuptools

setuptools.setup(
    name="active-reward-learning",
    author="David Lindner"
    version="0.1dev",
    description="",
    long_description=open("README.md").read(),
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "numpy>=1.16.1",
        "scipy==1.5.4",  # other versions lead to problems in optimizing highway
        "matplotlib",
        "gym==0.17.3",
        "sacred==0.8.2",
        "cvxopt",
        "networkx",
        "wrapt",
        "seaborn",
        "frozendict",
        "gast==0.2.2",
        "torch",
        "stable_baselines3",
        "tensorboard",
        "opencv-python",
    ],
    setup_requires=["pytest-runner"],
    extras_require={
        "interactive_environments": ["pygame"],
        "web_interface": ["flask"],
        "mobile_experiment_notifications": ["python-telegram-bot"],
    },
    tests_require=["pytest", "pytest-cov"],
    packages=setuptools.find_packages(),
    zip_safe=True,
    entry_points={},
    test_suite="active_reward_learning.tests",
)
