
from matplotlib import pyplot as plt
from tkinter.filedialog import askdirectory
from tkinter import Tk
import os

from utils import create_folder_data


def askdirectories(request_string: str = "Select folder", selected_directory_list: list = None, search_directory=None):
    if selected_directory_list is None:
        selected_directory_list = []
    directory_path_string = askdirectory(
        initialdir=search_directory, title=request_string)

    if len(directory_path_string) > 0:
        selected_directory_list.append(directory_path_string)
        askdirectories('Select the next Directory or Cancel to end',
                               selected_directory_list,
                               os.path.dirname(directory_path_string))

    return selected_directory_list


def open_step_by_step():
    folder = None
    while folder != "":
        folder = askdirectory()
        if folder != "":
            group_counts = create_folder_data(folder)
            x = group_counts['simTime']
            s = group_counts['group-s']
            s_name = 'susceptible ' + os.path.basename(folder)
            r = group_counts['group-r']
            r_name = 'recovered ' + os.path.basename(folder)
            i = group_counts['group-i']
            i_name = 'infected ' + os.path.basename(folder)
            plt.figure()
            plt.stackplot(x, i, s, r, labels=[i_name, s_name, r_name])
            plt.legend(loc='upper left')
    plt.show()


def open_multiple():
    folders = askdirectories()
    for folder in folders:
        group_counts = create_folder_data(folder)
        x = group_counts['simTime']
        s = group_counts['group-s']
        s_name = 'susceptible ' + os.path.basename(folder)
        r = group_counts['group-r']
        r_name = 'recovered ' + os.path.basename(folder)
        i = group_counts['group-i']
        i_name = 'infected ' + os.path.basename(folder)
        plt.figure()
        plt.stackplot(x, i, s, r, labels=[i_name, s_name, r_name])
        plt.legend(loc='upper left')
    plt.show()

def hardcoded_plot_1():
    fig, axs = plt.subplots(4, 3)
    folders = [
        ["C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.01_r0.01","C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.01_r0.02","C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.01_r0.05"],
        ["C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.02_r0.01","C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.02_r0.02","C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.02_r0.05"],
        ["C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.05_r0.01","C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.05_r0.02","C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.05_r0.05"],
        ["C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.1_r0.01","C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.1_r0.02","C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_p0.1_r0.05"],
    ]
    for idx, row in enumerate(folders):
        for idx2, folder in enumerate(row):
            group_counts = create_folder_data(folder)
            x = group_counts['simTime']
            s = group_counts['group-s']
            r = group_counts['group-r']
            i = group_counts['group-i']
            axs[idx, idx2].stackplot(x, i, s, r)

            axs[idx, idx2].set_title(folder.replace(
                "C:/GIT/Exercises/Exercise 2/output/Task5_test1_1000peds_","").replace("_"," "))

    for ax in axs.flat:
        ax.set(xlabel='Timesteps', ylabel='Pedestrians')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()


def hardcoded_plot_2():
    fig, axs = plt.subplots(2, 2)
    folders = [
        ["C:/GIT/Exercises/Exercise 2/output/Task5_supermarket_personalspace1.2",
         "C:/GIT/Exercises/Exercise 2/output/Task5_supermarket_personalspace2"],
        ["C:/GIT/Exercises/Exercise 2/output/Task5_supermarket_personalspace1.2_45peds",
         "C:/GIT/Exercises/Exercise 2/output/Task5_supermarket_personalspace2_45peds"],
    ]
    for idx, row in enumerate(folders):
        for idx2, folder in enumerate(row):
            group_counts = create_folder_data(folder)
            x = group_counts['simTime']
            s = group_counts['group-s']
            r = group_counts['group-r']
            i = group_counts['group-i']
            print(x)
            axs[idx, idx2].stackplot(x, i, s, r)

            axs[idx, idx2].set_title(folder.replace(
                "C:/GIT/Exercises/Exercise 2/output/Task5_supermarket_", "").replace("_", " "))

    for ax in axs.flat:
        ax.set(xlabel='Timesteps', ylabel='Pedestrians')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

root = Tk()
root.withdraw()
plt.ioff()  # Plots are shown blocking
# plt.show()

# open_step_by_step()
# open_multiple()

# hardcoded_plot_1()
hardcoded_plot_2()
