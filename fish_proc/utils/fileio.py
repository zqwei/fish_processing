def make_tarfile(output_filename, source_dir):
    import tarfile
    import os
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

        
def du(path):
    import subprocess
    # return subprocess.check_output(['du','-sh', path]).split()[0]
    return subprocess.check_output(['du', path]).split()[0]


def chmod(path, mode='0775'):
    import subprocess
    subprocess.call(['chmod', '-R', mode, path])
