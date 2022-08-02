import platform
import os
import subprocess
import shutil

IS_LINUX = "Linux" in platform.platform()

if not IS_LINUX:
    print("JAX currently only works under Linux so this software is currently limited to be Linux only")
    exit(-1)

install_deps = input("Should I run 'pip3 install -r requirements.txt' for you?\
 Accepting this is recommended only when working in a venv. \
Otherwise the installation of these packages will be global! [y/N] ")

if install_deps.lower() == "y":
    subprocess.run(["python3", "-m", "pip", "install", "-r", "requirements.txt"])

try:
    import FEMcommon.fem_toolkit
except Exception as ex:
    # check for rust toolchain
    s = subprocess.run(["cargo" ,"--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = s.stdout.decode("ascii")
    CARGO_INSTALLED = "cargo" in stdout
    
    # check for git
    s = subprocess.run(["git" ,"--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = s.stdout.decode("ascii")
    GIT_INSTALLED = "git" in stdout
    
    if not GIT_INSTALLED:
        print("Please install git!")
    
    if not CARGO_INSTALLED:
        print("Please install the Rust programming language toolchain by going to https://www.rust-lang.org/tools/install")
        exit(-1)
    else:
        compile = input("Missing Rust-based dynamic library. Should I compile it for you? This might take a while! [y/N] ")
        if compile.lower() == "y":
            subprocess.run(["git", "clone", "https://github.com/vasilvas99/fem-helpers.git"])
            os.chdir("fem-helpers")
            subprocess.run(["cargo", "build", "--release"])
            shutil.copyfile("./target/release/libfem_toolkit.so", "../FEMcommon/fem_toolkit.so")
            shutil.rmtree(os.getcwd())