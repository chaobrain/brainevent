import csv
import os
import shutil
import platform

linux_os = platform.system().lower() == "linux"

download_path = "./matrices/suitesparse"
os.makedirs(download_path, exist_ok=True)

filename = "0_matrix_list.csv"

total = sum(1 for line in open(filename))
print(total)

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for i in range(1, total):
        cur_row = next(csv_reader)
        matrix_group = f"{download_path}/" + cur_row[1]
        matrix_name = cur_row[2]
        if not os.path.exists(matrix_group + "/" + matrix_name + "/" + matrix_name + ".mtx"):
            os.makedirs(matrix_group, exist_ok=True)
            matrix_url = (
                "https://suitesparse-collection-website.herokuapp.com/MM/" +
                cur_row[1] + "/" + cur_row[2] + ".tar.gz"
            )
            print(matrix_url)
            # matrix_url = "http://sparse-files.engr.tamu.edu/MM/" + cur_row[1] + "/" + cur_row[2] + ".tar.gz"
            # os.system("axel -n 4 " + matrix_url)
            os.system(("wget " + matrix_url) if linux_os else ("curl " + matrix_url))
            shutil.move(matrix_name + ".tar.gz", download_path)
            os.system("tar -zxvf " + f"{download_path}/{matrix_name}" + ".tar.gz " + "-C " + matrix_group + "/")
            os.system("rm -rf " + f"{download_path}/{matrix_name}" + ".tar.gz")
