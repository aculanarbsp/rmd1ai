import yaml

with open("environment.yml") as file_handle:
    environment_data = yaml.load(file_handle, Loader=yaml.Loader)

input_string = "aiohappyeyeballs=2.4.4=pyhd8ed1ab_1"

# # Split the string on the '=' character
# parts = input_string.split('=')

# print(parts)
# Combine the first two parts to get "aiohappyeyeballs=2.4.4"
# result = '='.join(parts[:2])

# print(result)  # Output: aiohappyeyeballs=2.4.4

for dependency in environment_data["dependencies"]:
    print(dependency)

with open("requirements.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"]:
        
        if isinstance(dependency, dict):
            value_list = dependency.values()

            print(value_list)

            for value in list(value_list)[0]:
                file_handle.write(f"{value}\n")

            break

        package_name, package_version, extension = dependency.split("=")
        file_handle.write("{}=={} \n".format(package_name, package_version))

