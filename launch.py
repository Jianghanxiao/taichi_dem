import subprocess
import os
if __name__ == '__main__':
    # ger current working directory
    cwd = os.getcwd()
    script_dir = os.path.dirname(__file__)
    if(not os.path.samefile(cwd, script_dir)):
        print("> the current working directory is not the same as the script directory.\n"
              "> so we change the working directory to the script directory:\n> "
              + script_dir)
        # change directory to the folder where this file is located
        os.chdir(script_dir)
    
    print("> run the dem simulation")
    retcode = subprocess.run(["python", "dem/dem.py"])
    print("> dem simulation finished")
    
    print("> double click the file 'taichi_dem/visualizer.hipnc' to visualize the result in houdini")
    print("> download: https://www.sidefx.com/products/houdini")