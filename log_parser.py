import os
import pandas as pd
import matplotlib.pyplot as plt

#############################################
# Takes a directory of generation directories
# and parses the .log files and plots the box
# plots of each generation
#############################################


def log_parser(directory):
    box_plot_list = []
    folders = os.walk(directory)
    for folder in folders:
        #print(folder)
        for file in folder[2]:
            if file != '.DS_Store' and '.log' in file and 'parent' not in file:
                #print(file)
                f = open(folder[0] + '/' + file, 'r')
                line_list = f.readlines()
                if len(line_list) != 0:
                    if 'Score: ' in line_list[-1]:
                        box_plot_list.append([folder[0][-4:], float(line_list[-1][7:-1])])
                    else:
                        box_plot_list.append([folder[0][-4:], float(line_list[-1][0:-1])])

    # Turn the list of [[gen, score]] into a data frame and plot it as a box plot
    df = pd.DataFrame(box_plot_list)
    df.columns = ['Generation', 'Score']
    print(df)
    df.boxplot(column='Score', by='Generation')
    plt.show()


# Change the argument to this line so that it points to the correct directory

log_parser("C:/Users/nkala\Desktop\MQP\hmd_auth_dev/bvp_auth_project-code/bvp_genetic\genetic_5")


