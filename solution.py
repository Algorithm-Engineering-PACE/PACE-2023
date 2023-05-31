import subprocess
import sys

def install_packages():
    """Install packages from requirements.txt"""
    with open('requirements.txt') as f:
        packages = f.readlines()

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main(command):
  
    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Check the return code
    if process.returncode == 0:
        print(stdout.decode('utf-8'))
    else:
        print(f"Failed running main.py -  Error: {stderr.decode('utf-8')}")

if __name__=='__main__':
    # Ensure packages are installed
    #install_packages()
    if len(sys.argv) > 1:
        api_command = sys.argv[1]
        if api_command == "clean-results":
            command = ['python3', 'main.py', api_command]
        else:
            file_name = sys.argv[2]
            command = ['python3', 'main.py', api_command,file_name]
    else:
        command = ['python3', 'main.py', 'proccess-graph-from-input']
    main(command)
    
