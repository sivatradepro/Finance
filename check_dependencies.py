import pkg_resources

def list_installed_packages():
    installed_packages = []
    for package in pkg_resources.working_set:
        installed_packages.append((package.project_name, package.version))
    return installed_packages

if __name__ == "__main__":
    print("Installed Packages:")
    for package_name, package_version in list_installed_packages():
        print(f"{package_name}=={package_version}")
